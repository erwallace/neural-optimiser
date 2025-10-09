from typing import Any

import numpy as np
import torch
from ase import Atoms
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Draw, rdDetermineBonds
from torch_geometric.data import Data


class Conformer(Data):
    atom_type_dtype = torch.int64
    pos_dtype = torch.float32

    def __init__(
        self,
        atom_types: torch.Tensor,  # [n_atoms]
        pos: torch.Tensor,  # [n_atoms, 3]
        smiles: str | None = None,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.atom_types = atom_types
        self.pos = pos
        self.smiles = smiles

        for key, item in kwargs.items():
            setattr(self, key, item)

        self.to(device)
        self.__post_init__()

    def __post_init__(self):
        """Validate attributes."""
        if self.atom_types is None and self.pos is None:
            return  # allow PyG to construct placeholder during collate
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
    def n_atoms(self) -> int:
        """Number of atoms in the conformer."""
        return self.pos.size(0)

    @property
    def n_conformers(self) -> int:
        """Number of conformers (needed for Optimiser compatibility)."""
        return 1

    @classmethod
    def from_ase(cls, atoms: Atoms, **kwargs) -> "Conformer":
        """Construct Conformer from ASE Atoms object."""
        z = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        pos = np.asarray(atoms.get_positions(), dtype=np.float32)

        if "charge" not in kwargs:
            kwargs["charge"] = atoms.info.get("charge", 0)
        if "spin" not in kwargs:
            kwargs["spin"] = atoms.info.get("spin", 1)

        return cls(
            atom_types=torch.from_numpy(z),
            pos=torch.from_numpy(pos),
            **kwargs,
        )

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol, conf: Chem.Conformer | None = None, **kwargs) -> "Conformer":
        """Construct Conformer from RDKit Mol with 3D conformer or Conformer."""
        if mol is None:
            raise ValueError("mol is None")
        n = mol.GetNumAtoms()
        if n == 0:
            raise ValueError("mol has no atoms")

        if conf is None:
            n_confs = mol.GetNumConformers()
            if n_confs == 0:
                raise ValueError("RDKit Mol has no conformers. Provide a 3D conformer.")
            elif n_confs > 1:
                logger.warning(
                    f"RDKit Mol has {n_confs} conformers. Using the first conformer (ID 0)."
                )
            conf = mol.GetConformer()

        # Atomic numbers
        z = np.fromiter((a.GetAtomicNum() for a in mol.GetAtoms()), count=n, dtype=np.int64)
        # Positions
        pos = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            pos[i] = (float(p.x), float(p.y), float(p.z))

        if "charge" not in kwargs:
            kwargs["charge"] = Chem.GetFormalCharge(mol)
        if "spin" not in kwargs:
            kwargs["spin"] = hunds_rule(mol)
        if "smiles" not in kwargs:
            kwargs["smiles"] = Chem.MolToSmiles(mol)

        return cls(
            atom_types=torch.from_numpy(z),
            pos=torch.from_numpy(pos),
            **kwargs,
        )

    def to_ase(self) -> Atoms:
        """Convert to ASE Atoms."""
        numbers = self.atom_types.detach().cpu().numpy().astype(int)
        positions = self.pos.detach().cpu().numpy().astype(np.float32)
        atoms = Atoms(numbers=numbers, positions=positions)

        for k, v in self.__dict__["_store"].items():
            if not isinstance(v, torch.Tensor):
                atoms.info[k] = v

        return atoms

    def to_rdkit(self) -> Chem.Mol:
        """Convert to an RDKit Mol with a single 3D conformer (no bonds)."""
        from rdkit.Geometry import Point3D  # local import to avoid top-level dependency issues

        z = self.atom_types.detach().cpu().numpy().astype(int)
        pos = self.pos.detach().cpu().numpy().astype(np.float32)

        rw = Chem.RWMol()
        for Zi in z:
            rw.AddAtom(Chem.Atom(int(Zi)))

        mol = rw.GetMol()

        conf = Chem.Conformer(len(z))
        conf.Set3D(True)
        for i, (x, y, zc) in enumerate(pos):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(zc)))
        mol.AddConformer(conf, assignId=True)

        for k, v in self.__dict__["_store"].items():
            if not isinstance(v, torch.Tensor):
                mol.SetProp(k, str(v))

        return mol

    def _plot_2d(self) -> None:
        """Plot 2D structure from SMILES string."""
        if self.smiles is None:
            raise ValueError("Cannot plot 2D structure without SMILES string.")
        mol = Chem.MolFromSmiles(self.smiles)
        return Draw.MolToImage(mol)

    def _plot_3d(self) -> None:
        """Plot 3D structure from atomic positions."""
        mol = self.to_rdkit()
        try:
            rdDetermineBonds.DetermineBonds(mol)
        except Exception as e:
            raise ValueError("Could not determine bonds for 3D plot.") from e
        return Draw.MolToImage(mol)

    def plot(self, dim: int = 2) -> None:
        """Plot the conformer in 2D or 3D."""
        if dim == 2:
            return self._plot_2d()
        elif dim == 3:
            return self._plot_3d()
        else:
            raise ValueError("dim must be 2 or 3")


def hunds_rule(mol: Chem.Mol) -> int:
    """Calculate spin multiplicity using Hund's rule."""
    num_radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    total_electronic_spin = num_radical_electrons / 2
    spin_multiplicity = 2 * total_electronic_spin + 1
    return int(spin_multiplicity)


if __name__ == "__main__":
    from ase.build import molecule
    from rdkit.Chem import AllChem

    # Example usage: RDKit
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    conformer = Conformer.from_rdkit(mol)
    print(conformer)
    d2 = conformer.plot(dim=2)
    # d2.show()
    d3 = conformer.plot(dim=3)
    # d3.show()

    conformer2 = Conformer.from_rdkit(mol, charge=4)
    print(conformer2.charge)

    # Example usage: ASE
    atoms = molecule("H2O")
    conformer_ase = Conformer.from_ase(atoms)
    print(conformer_ase)
    d3 = conformer_ase.plot(dim=3)
    # d3.show()
