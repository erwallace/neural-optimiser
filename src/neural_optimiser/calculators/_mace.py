import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

from neural_optimiser.calculators.base import Calculator


class MACECalculator(Calculator):
    """Calculator using a MACE model for energy and force predictions."""

    def __init__(self, model_paths: str, device: str = "cpu", max_neighbours: int = 32):
        try:
            from mace.tools.utils import AtomicNumberTable
        except ImportError:
            raise ImportError("MACE is not installed. Run `uv pip install mace-torch` to install.")

        try:  # needed for radius_graph
            import torch_cluster  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch-cluster is not installed. Run `uv pip install torch-cluster` to install."
            )

        self.device = device
        self.max_neighbours = max_neighbours
        self.model_paths = model_paths
        self.model = torch.load(f=model_paths, map_location=device, weights_only=False)
        self.model.requires_grad_(False).eval().to(device)

        self._z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

    def __repr__(self) -> str:
        return (
            f"MACECalculator(model_paths={self.model_paths}, device={self.device}, "
            f"max_neighbours={self.max_neighbours})"
        )

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces for a batch of conformers using the MACE model."""
        atomic_data = self.to_atomic_data(batch)
        output = self.model(atomic_data, compute_force=True)
        return output["energy"], output["forces"]

    def get_energies(self, batch: Data | Batch) -> torch.Tensor:
        """Compute energies for a batch of conformers using the MACE model."""
        atomic_data = self.to_atomic_data(batch)
        output = self.model(atomic_data, compute_force=False)
        return output["energy"]

    def to_atomic_data(
        self,
        batch: Data | Batch,
    ) -> Batch:
        """Convert a ConformerBatch into a torch_geometric Batch[Data] with MACE fields."""
        self._validate_batch(batch)

        if batch.pos.numel() == 0:
            return Batch()

        model_dtype = next(self.model.parameters()).dtype

        # Node features: one-hot encodings from atomic numbers
        node_indices = self.atomic_numbers_to_indices(
            batch.atom_types, z_table=self._z_table
        )  # [N]
        one_hot_atoms = self.to_one_hot(node_indices, num_classes=len(self._z_table))  # [N, Z]

        # Graph partition info
        n_graphs = batch.n_conformers
        num_atoms_per_graph = torch.bincount(batch.batch, minlength=n_graphs)  # [G]
        cumsum = torch.cumsum(num_atoms_per_graph, dim=0)  # [G]
        starts = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.device), cumsum[:-1]], dim=0
        )  # [G]

        # Neighborhoods for full batch
        full_edge_index = radius_graph(
            x=batch.pos,
            r=float(self.model.r_max),
            batch=batch.batch,
            loop=False,
            max_num_neighbors=self.max_neighbours,
        )  # [2, E]
        edge_graph = batch.batch[full_edge_index[0]]

        # Per-graph slices
        split_points = cumsum[:-1].tolist()
        coords_per_graph = torch.tensor_split(batch.pos, split_points, dim=0)
        feats_per_graph = torch.tensor_split(one_hot_atoms, split_points, dim=0)

        head_tensor = self._resolve_head_tensor(device=self.device)

        data_list: list[Data] = []
        for g in range(n_graphs):
            coords_g = coords_per_graph[g]
            feats_g = feats_per_graph[g].to(dtype=model_dtype)

            # Edge slice and local reindexing
            mask_g = edge_graph == g
            edge_index_g = full_edge_index[:, mask_g]
            local_edge_index = edge_index_g - starts[g]
            e_num = local_edge_index.size(1)

            # Shifts/cell (no PBC)
            zeros_e3 = torch.zeros((e_num, 3), dtype=model_dtype, device=self.device)
            zero_cell = torch.zeros((3, 3), dtype=model_dtype, device=self.device)

            # Edge vectors/lengths
            if e_num:
                src, dst = local_edge_index[0], local_edge_index[1]
                edge_vectors = coords_g[dst] - coords_g[src]
                edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)
            else:
                edge_vectors = torch.zeros((0, 3), dtype=coords_g.dtype, device=self.device)
                edge_lengths = torch.zeros((0,), dtype=coords_g.dtype, device=self.device)

            data_list.append(
                Data(
                    edge_index=local_edge_index,
                    node_attrs=feats_g,
                    positions=coords_g,
                    shifts=zeros_e3,
                    unit_shifts=zeros_e3.clone(),
                    cell=zero_cell,
                    head=head_tensor,
                    edge_vectors=edge_vectors,
                    edge_lengths=edge_lengths,
                )
            )

        return Batch.from_data_list(data_list)

    def _resolve_head_tensor(self, device: str) -> torch.Tensor:
        heads = self.model.heads
        if heads is None:
            heads = ["Default"]
        elif isinstance(heads, str):
            heads = [heads]

        head_name = heads[0] if len(heads) == 1 else heads[-1]
        head_idx = heads.index(head_name) if head_name in heads else len(heads) - 1
        return torch.tensor(head_idx, dtype=torch.long, device=device)

    @staticmethod
    def _validate_batch(batch: Data | Batch) -> None:
        required = ("pos", "batch", "atom_types")
        missing = [name for name in required if not hasattr(batch, name)]
        if missing:
            raise AttributeError(
                f"Batch must have attributes: {', '.join(required)}. Missing: {missing}"
            )

    @staticmethod
    def atomic_numbers_to_indices(
        atomic_numbers_tensor: torch.Tensor,
        z_table: "AtomicNumberTable",  # noqa: F821
    ) -> torch.Tensor:
        """
        Map atomic numbers to z_table indices using a vectorized torch operation.

        :param atomic_numbers_tensor: Integer tensor of atomic numbers (any shape).
        :param z_table: AtomicNumberTable providing z_to_index(Z) -> index and len(z_table).
        :return: Tensor of same shape as atomic_numbers_tensor with z_table indices (long).
        """
        if atomic_numbers_tensor.dtype not in (torch.int32, torch.int64, torch.long):
            raise TypeError("atomic_numbers_tensor must be an integer dtype.")

        device = atomic_numbers_tensor.device
        unique_z = torch.unique(atomic_numbers_tensor)
        max_z = int(unique_z.max().item())

        # Build dense LUT of size (max_z + 1), default -1 for missing entries.
        lut = torch.full((max_z + 1,), -1, dtype=torch.long, device=device)
        for z in unique_z.tolist():
            lut[z] = int(z_table.z_to_index(z))

        mapped = lut[atomic_numbers_tensor]
        if (mapped < 0).any():
            missing = atomic_numbers_tensor[mapped < 0].unique().tolist()
            raise ValueError(f"Found atomic numbers not in z_table: {missing}")

        return mapped

    @staticmethod
    def to_one_hot(
        indices: torch.Tensor,
        num_classes: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Convert integer indices to one-hot encodings with broadcasting support.

        :param indices: Integer tensor of indices (...,) or (..., 1).
        :param num_classes: Number of classes for one-hot dimension.
        :param dtype: Optional torch dtype for result. Defaults to torch.get_default_dtype().
        :return: One-hot tensor of shape (..., num_classes) with float dtype by default.
        """
        if indices.dim() >= 1 and indices.size(-1) == 1:
            indices = indices.squeeze(-1)
        one_hot = F.one_hot(indices, num_classes=num_classes)
        result_dtype = dtype or torch.get_default_dtype()
        return one_hot.to(dtype=result_dtype)


if __name__ == "__main__":
    from ase.build import molecule

    from neural_optimiser.conformers import ConformerBatch

    atoms = molecule("H2O")
    batch = ConformerBatch.from_ase([atoms], device="cpu")

    model_paths = "./models/MACE_SPICE2_NEUTRAL.model"
    calculator = MACECalculator(model_paths=model_paths, device="cpu")

    e, f = calculator(batch)

    from mace.calculators.mace import MACECalculator as MACECalc

    mace_calc = MACECalc(model_paths=model_paths, device="cpu")
    atoms.calc = mace_calc
    _e = atoms.get_potential_energy()
    _f = atoms.get_forces()

    assert torch.allclose(e, torch.tensor(_e), atol=1e-4)
    assert torch.allclose(f, torch.tensor(_f, dtype=torch.float32), atol=1e-4)
