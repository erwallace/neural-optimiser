import torch
import torch.nn.functional as F
from mace.tools.utils import AtomicNumberTable
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

from neural_optimiser.calculators.base import Calculator


class MACECalculator(Calculator):
    def __init__(self, model_paths: str, device: str = "cpu"):
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

        self.model = torch.load(f=model_paths, map_location=device, weights_only=False)
        self.device = device

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.model.to(device)

        self._z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        model_dtype = next(self.model.parameters()).dtype

        batch = AtomicData.from_conformer_batch(
            batch=batch,
            model_dtype=model_dtype,
            z_table=self._z_table,
            cutoff=float(self.model.r_max),
            heads=self.model.heads,
        )

        # for k,v in batch.__dict__["_store"].items():
        #     print(k, v.dtype, v.shape)
        #     if v.dtype == torch.float32:
        #         batch.__dict__["_store"][k] = v.to(dtype=model_dtype)

        output = self.model(batch, compute_force=True)
        return output

    def to_atomic_data():
        pass

    def from_atomic_data():
        pass


def atomic_numbers_to_indices(
    atomic_numbers_tensor: torch.Tensor,
    z_table: AtomicNumberTable,
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


class AtomicData(Data):
    """
    Lightweight Data container with edges computed via torch_geometric.nn.radius_graph.
    No periodic boundary conditions are assumed, and default zero cell/shifts are used.

    :param edge_index: Edge index tensor [2, E].
    :param node_attrs: Node feature tensor [N, F].
    :param positions: Node position tensor [N, 3].
    :param shifts: Edge shift vectors [E, 3]; defaults to zeros when None.
    :param unit_shifts: Unit shift vectors [E, 3]; defaults to zeros when None.
    :param cell: Cell matrix [3, 3]; defaults to zeros when None.
    :param head: Scalar long tensor indicating head index.
    :param edge_vectors: Optional precomputed edge vectors [E, 3].
    :param edge_lengths: Optional precomputed edge lengths [E].
    """

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor | None,  # [n_edges, 3], zeros when no PBC
        unit_shifts: torch.Tensor | None,  # [n_edges, 3], zeros when no PBC
        cell: torch.Tensor | None,  # [3,3], zeros matrix
        head: torch.Tensor | None,  # scalar (long)
        edge_vectors: torch.Tensor | None = None,  # [n_edges, 3]
        edge_lengths: torch.Tensor | None = None,  # [n_edges]
    ):
        if positions is not None and edge_index is not None and node_attrs is not None:
            if positions.shape[-1] != 3:
                raise ValueError("positions must have shape [N, 3].")
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                raise ValueError("edge_index must have shape [2, E].")
            if node_attrs.dim() != 2 or node_attrs.size(0) != positions.size(0):
                raise ValueError("node_attrs must be [N, F] and align with positions.")

            device = positions.device
            ptdtype = positions.dtype

            shifts = (
                shifts
                if shifts is not None
                else torch.zeros(edge_index.size(1), 3, dtype=ptdtype, device=device)
            )
            unit_shifts = (
                unit_shifts
                if unit_shifts is not None
                else torch.zeros(edge_index.size(1), 3, dtype=ptdtype, device=device)
            )
            cell = cell if cell is not None else torch.zeros(3, 3, dtype=ptdtype, device=device)

            if edge_vectors is None:
                src, dst = edge_index[0], edge_index[1]
                edge_vectors = positions[dst] - positions[src]
            if edge_lengths is None:
                edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)

        super().__init__(
            edge_index=edge_index,
            node_attrs=node_attrs,
            positions=positions,
            shifts=shifts,
            unit_shifts=unit_shifts,
            cell=cell,
            head=head,
            edge_vectors=edge_vectors,
            edge_lengths=edge_lengths,
        )

    @classmethod
    def from_conformer_batch(
        cls,
        batch: Data | Batch,
        model_dtype: torch.dtype,
        z_table: AtomicNumberTable,
        cutoff: float,
        max_num_neighbors: int | None = 32,
        heads: list[str] | str | None = None,
    ) -> Batch:
        """
        Convert a batched graph into individual AtomicData objects with neighbor information.

        :param batch: Graph batch. Requires attributes .pos [N, 3], .atom_type [N], and .batch [N]
        :param cutoff: Radius cutoff for neighbor search.
        :param max_num_neighbors: Maximum neighbors per node for performance/footprint control.
        :param heads: List of head names or a single head string. Unrecognized heads map to last.
        :return: torch_geometric.data.Batch containing ConformerBatch graphs on the original device.
        """
        if heads is None:
            heads = ["Default"]
        elif isinstance(heads, str):
            heads = [heads]

        if hasattr(batch, "pos") and hasattr(batch, "batch") and hasattr(batch, "atom_types"):
            cls.device = batch.pos.device
        else:
            raise AttributeError("Batch must have `pos`, `atom_types` and `batch` attributes.")

        # Handle empty batch
        if batch.pos.numel() == 0:
            return Batch()

        node_indices_all = atomic_numbers_to_indices(batch.atom_types, z_table=z_table)  # [N]
        one_hot_atoms = to_one_hot(node_indices_all, num_classes=len(z_table))  # [N, Z]

        # Graph split information
        n_graphs = batch.n_conformers
        num_atoms_per_graph = torch.bincount(batch.batch, minlength=n_graphs)  # [N]
        cumsum = torch.cumsum(num_atoms_per_graph, dim=0)  # [N]
        starts = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=cls.device), cumsum[:-1]], dim=0
        )  # [G]

        # Build neighborhoods for full batch
        full_edge_index = radius_graph(
            x=batch.pos,
            r=cutoff,
            batch=batch.batch,
            loop=False,
            max_num_neighbors=max_num_neighbors,
        )  # [2, E]

        # Edge graph ids (radius_graph respects batch, but used for slicing)
        edge_graph = batch.batch[full_edge_index[0]]

        # Per-graph slices
        split_points = cumsum[:-1].tolist()
        coords_list = torch.tensor_split(batch.pos, split_points, dim=0)
        one_hot_list = torch.tensor_split(one_hot_atoms, split_points, dim=0)

        # Head index resolution
        head_name = heads[0] if len(heads) == 1 else heads[-1]
        head_idx = heads.index(head_name) if head_name in heads else len(heads) - 1
        head_tensor = torch.tensor(head_idx, dtype=torch.long, device=cls.device)

        data_list: list[AtomicData] = []
        for g in range(n_graphs):
            # Node slices
            coords_g = coords_list[g]
            feats_g = one_hot_list[g]

            # Edge slice and local reindexing
            mask_g = edge_graph == g
            edge_index_g = full_edge_index[:, mask_g]
            local_edge_index = edge_index_g - starts[g]
            e_num = local_edge_index.size(1)

            zeros_e3 = torch.zeros(e_num, 3, dtype=torch.get_default_dtype(), device=cls.device)
            zero_cell = torch.zeros(3, 3, dtype=torch.get_default_dtype(), device=cls.device)

            feats_g = feats_g.to(dtype=model_dtype)

            data_g = cls(
                edge_index=local_edge_index,
                node_attrs=feats_g,
                positions=coords_g,
                shifts=zeros_e3,
                unit_shifts=zeros_e3.clone(),
                cell=zero_cell,
                head=head_tensor,
            )
            data_list.append(data_g)

        return Batch.from_data_list(data_list).to(cls.device)


if __name__ == "__main__":
    from ase.build import molecule

    from neural_optimiser.conformers import ConformerBatch

    atoms = molecule("H2O")
    batch = ConformerBatch.from_ase([atoms], device="cpu")

    model_paths = "./models/MACE_SPICE2_NEUTRAL.model"
    calculator = MACECalculator(model_paths=model_paths, device="cpu")

    out = calculator.calculate(batch)
    e, f = out["energy"], out["forces"]

    from mace.calculators.mace import MACECalculator as MACECalc

    mace_calc = MACECalc(model_paths=model_paths, device="cpu")
    atoms.calc = mace_calc
    _e = atoms.get_potential_energy()
    _f = atoms.get_forces()

    assert torch.allclose(e, torch.tensor(_e), atol=1e-4)
    assert torch.allclose(f, torch.tensor(_f, dtype=torch.float32), atol=1e-4)
