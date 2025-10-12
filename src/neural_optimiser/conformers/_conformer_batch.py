import torch
from ase import Atoms
from rdkit import Chem
from torch_geometric.data import Batch

from neural_optimiser.conformers import Conformer


class ConformerBatch(Batch):
    """A batch of molecular conformers."""

    atom_type_dtype = torch.int64
    pos_dtype = torch.float32
    molecule_idx_dtype = torch.int64

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.__post_init__()
        self.to(self.device)

    def __post_init__(self):
        """Validate attributes."""

        if getattr(self, "atom_types", None) is None or getattr(self, "pos", None) is None:
            return  # placeholder instance during Batch.from_data_list
        if self.pos.ndim != 2 or self.pos.size(-1) != 3:
            raise ValueError(f"pos must have shape [n_atoms, 3], got {tuple(self.pos.shape)}")
        if self.atom_types.ndim != 1:
            raise ValueError(
                f"atom_types must be 1-D [n_atoms], got {tuple(self.atom_types.shape)}"
            )
        if self.atom_types.size(0) != self.pos.size(0):
            raise ValueError(
                f"atom_types and pos must have matching n_atoms, "
                f"got {self.atom_types.size(0)} vs {self.pos.size(0)}"
            )
        if self.atom_types.dtype != self.atom_type_dtype:
            raise ValueError(
                f"atom_types must have dtype {self.atom_type_dtype}, got {self.atom_types.dtype}"
            )
        if self.pos.dtype != self.pos_dtype:
            raise ValueError(f"pos must have dtype {self.pos_dtype}, got {self.pos.dtype}")

    @property
    def n_molecules(self) -> int:
        """Number of molecules in the batch."""
        return len(set(self.smiles)) if hasattr(self, "smiles") else None

    @property
    def n_conformers(self) -> int:
        """Number of conformers in the batch."""
        return self.batch.max().item() + 1

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the batch."""
        return self.pos.size(0)

    def conformer(self, idx: int, step: int | None = None) -> Conformer:
        """Get the idx-th conformer in the batch at the n-th relaxation step."""
        kwargs = {}

        for k, v in self.__dict__["_store"].items():
            if torch.is_tensor(v) and v.size(0) == self.n_atoms:
                kwargs[k] = v[self.batch == idx]
            elif torch.is_tensor(v) and v.size(0) == self.n_conformers:
                kwargs[k] = v[idx]

        if step is not None:  # if optimisation has been performed
            if hasattr(self, "pos_dt"):
                kwargs["pos"] = self.pos_dt[step][self.batch == idx]
                kwargs["forces"] = self.forces_dt[step][self.batch == idx]
                kwargs["energy"] = self.energies_dt[step, idx]
            else:
                raise ValueError(f"Cannot return step {step}, no pos_dt attribute found in batch.")

        return Conformer(**kwargs)

    @classmethod
    def from_data_list(cls, data_list: list, device: str = "cpu", *args, **kwargs):
        """Wrap Batch.from_data_list to finalize attributes."""
        batch = super().from_data_list(data_list, *args, **kwargs)
        batch.to(device=device)
        batch.__post_init__()
        return batch

    @classmethod
    def from_rdkit(
        cls, mol: list[Chem.Mol] | Chem.Mol, device: str = "cpu", **kwargs
    ) -> "ConformerBatch":
        """Create a ConformerBatch from a list of RDKit Mol objects.

        Each Mol can have multiple conformers.
        """
        if isinstance(mol, Chem.Mol):
            mol = [mol]

        conformers = []
        for i, mol in enumerate(mol):
            for conformer in mol.GetConformers():
                conformers.append(Conformer.from_rdkit(mol, conformer, **kwargs))

        batch = cls.from_data_list(conformers, device=device)
        return batch

    @classmethod
    def from_ase(cls, atoms_list: list[Atoms], device: str = "cpu") -> "ConformerBatch":
        """Create a ConformerBatch from a list of ASE Atoms objects."""
        conformers = [Conformer.from_ase(a) for a in atoms_list]
        return cls.from_data_list(conformers, device=device)

    def to_rdkit(self):
        """Convert each conformer in the batch to an RDKit Mol object."""
        return [self.conformer(i).to_rdkit() for i in range(self.n_conformers)]

    def to_ase(self):
        """Convert each conformer in the batch to an ASE Atoms object."""
        return [self.conformer(i).to_ase() for i in range(self.n_conformers)]


if __name__ == "__main__":
    from ase.build import molecule
    from rdkit.Chem import AllChem

    atoms_list = [molecule("H2O"), molecule("NH3")]
    batch = ConformerBatch.from_ase(atoms_list)
    print(batch)

    confs = [Conformer.from_ase(conf) for conf in atoms_list]
    batch = ConformerBatch.from_data_list(confs)
    print(batch)

    mols_list = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]
    for i, mol in enumerate(mols_list):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mols_list[i] = mol
    new_conf = Chem.Conformer(mols_list[0].GetConformer(0))
    mols_list[0].AddConformer(new_conf)
    batch = ConformerBatch.from_rdkit(mols_list)
    print(batch)

    print(batch.batch)

    print(batch.conformer(0))
