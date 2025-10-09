import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import radius_graph

from neural_optimiser.calculators.base import Calculator


class FAIRChemCalculator(Calculator):
    def __init__(
        self, model_paths: str, device: str = "cpu", radius: float = 6.0, max_neighbours: int = 32
    ):
        try:
            import fairchem  # noqa: F401
            from fairchem.core.datasets.atomic_data import (  # noqa: F401
                AtomicData,
                atomicdata_list_to_batch,
            )
        except ImportError:
            raise ImportError(
                "MACE is not installed. Run `uv pip install fairchem-core` to install."
            )
        self.device = device
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.predictor = load_predict_unit(path=model_paths, device=device)

        print(self.predictor.__dict__)

        # model_checkpoint = torch.load(f=model_paths, map_location=device, weights_only=False)
        # self.model = model_checkpoint.model
        # self.model.eval().to(device)

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces for a batch of conformers using a FAIRChem model."""
        # atomic_data = self.to_atomic_data(batch)
        atomic_data = self.non_batched_to_atomic_data(batch)
        output = self.predictor.predict(atomic_data)  # TODO: atomic data not correctly formatted
        return output["energy"], output["forces"]

    def get_energies(self, batch: Data | Batch) -> torch.Tensor:
        raise NotImplementedError("FAIRChemCalculator is not yet implemented.")

    def non_batched_to_atomic_data(self, batch: Data | Batch) -> Batch:
        atomic_data_list = []
        for i in range(batch.n_conformers):
            conformer = batch.conformer(i)
            ase_atoms = conformer.to_ase()
            atomic_data = AtomicData.from_ase(
                ase_atoms, r_edges=True, radius=self.radius, max_neigh=self.max_neighbours
            )
            atomic_data_list.append(atomic_data)

        return atomicdata_list_to_batch(atomic_data_list)

    def to_atomic_data(
        self,
        batch: Data | Batch,
    ) -> Batch:
        """
        Refactored batched converter to match the MACE architecture's data preprocessing.
        This version uses torch_geometric.nn.radius_graph for efficient, batch-wise
        graph construction.

        Args:
            batch (ConformerBatch): The batch of conformers to convert.

        Returns:
            AtomicData: A single batched AtomicData object ready for fairchem models.
        """
        n_conformers = batch.n_conformers
        model_dtype = batch.pos.dtype

        # 1. Batched graph construction using radius_graph
        edge_index = radius_graph(
            x=batch.pos,
            r=self.radius,
            batch=batch.batch,
            loop=False,
            max_num_neighbors=self.max_neighbours,
        )

        # 2. Create other AtomicData fields in a batched manner
        # For non-periodic systems (molecules), cell_offsets are all zero.
        cell_offsets = torch.zeros((edge_index.shape[1], 3), dtype=model_dtype, device=self.device)

        # `natoms` is the count of atoms per graph
        natoms = torch.bincount(batch.batch, minlength=n_conformers).to(self.device)

        # `nedges` is the count of edges per graph
        # We find which graph each edge belongs to by looking at the batch index of the source node
        edge_batch = batch.batch[edge_index[0]]
        nedges = torch.bincount(edge_batch, minlength=n_conformers).to(self.device)

        # Create dummy cells and PBC flags as required by AtomicData for non-periodic molecules
        cell = torch.zeros((n_conformers, 3, 3), dtype=model_dtype, device=self.device)
        pbc = torch.zeros((n_conformers, 3), dtype=torch.bool, device=self.device)

        # Other properties
        charge = getattr(
            batch, "charge", torch.zeros(n_conformers, dtype=torch.long, device=self.device)
        )
        # if charge.dim() == 0:
        #     charge = charge.unsqueeze(0)

        spin = getattr(
            batch, "spin", torch.ones(n_conformers, dtype=torch.long, device=self.device)
        )
        # if spin.dim() == 0:
        #     spin = spin.unsqueeze(0)

        fixed = torch.zeros_like(batch.atom_types)
        tags = torch.zeros_like(batch.atom_types)

        # 3. Assemble the final AtomicData object
        atomic_data_batch = AtomicData(
            pos=batch.pos,
            atomic_numbers=batch.atom_types,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=charge,
            spin=spin,
            fixed=fixed,
            tags=tags,
            batch=batch.batch,
            sid=[f"conformer_{i}" for i in range(n_conformers)],
            dataset=["s2ef" for _ in range(n_conformers)],
        )

        return atomic_data_batch


if __name__ == "__main__":
    from ase.build import molecule
    from fairchem.core import FAIRChemCalculator as FAIRChemCalc
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from fairchem.core.units.mlip_unit import load_predict_unit

    from neural_optimiser.conformers import ConformerBatch

    atoms = molecule("H2O")
    batch = ConformerBatch.from_ase([atoms, atoms], device="cpu")

    model_paths = "./models/omol25_esen_md_direct.pt"
    device = "cpu"

    predictor = load_predict_unit(path=model_paths, device=device)
    fairchem_calc = FAIRChemCalc(predictor, task_name="omol")
    atoms.calc = fairchem_calc
    print("fairchem calc set up")
    _e = atoms.get_potential_energy()
    _f = atoms.get_forces()
    print(_e, _f)

    my_calulator = FAIRChemCalculator(model_paths=model_paths, device=device)
    print("my calc set up")
    e, f = my_calulator(batch)
    print(e, f)

    assert torch.allclose(e, torch.tensor(_e), atol=1e-4)
    assert torch.allclose(f, torch.tensor(_f, dtype=torch.float32), atol=1e-4)
