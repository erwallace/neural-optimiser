from collections.abc import Mapping
from typing import Any

import torch
from ase.units import eV, kcal, mol
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator

KCAL_PER_MOL_TO_EV = kcal / mol / eV


class MMFF94Calculator(Calculator):
    """Calculator using RDKit's implementation of the Merck Molecular Force Field (MMFF94(s))."""

    def __init__(
        self,
        MMFFGetMoleculeProperties: Mapping | None = None,
        MMFFGetMoleculeForceField: Mapping | None = None,
    ):
        self.mol_prop = dict(MMFFGetMoleculeProperties or {})
        self.mol_ff = dict(MMFFGetMoleculeForceField or {})
        self.bond_info = None
        self._cache_key = None

    def __repr__(self) -> str:
        parts = []
        if self.mol_prop:
            parts.extend(f"{k}={v}" for k, v in self.mol_prop.items())
        if self.mol_ff:
            parts.extend(f"{k}={v}" for k, v in self.mol_ff.items())
        args = ", ".join(parts)
        return f"MMFFCalculator({args})" if args else "MMFFCalculator()"

    def _ensure_single_conformer(self, batch: Data | Batch) -> None:
        """Ensure that the batch contains only a single conformer."""
        if isinstance(batch, Batch) and batch.batch.unique().size(0) != 1:
            raise ValueError("MMFFCalculator only supports single-conformer batches.")

    def _to_mol(self, batch: Data | Batch) -> Chem.RWMol:
        """Convert a single-conformer batch to an RDKit RWMol."""
        if isinstance(batch, Batch):
            return Chem.RWMol(batch.to_rdkit()[0])
        return Chem.RWMol(batch.to_rdkit())

    def _get_charge(self, mol: Chem.Mol) -> int:
        """Get the formal charge of the molecule from its properties, defaulting to 0."""
        return int(mol.GetProp("charge")) if mol.HasProp("charge") else 0

    def _molecule_key(self, mol: Chem.Mol) -> str:
        """Generate a unique key for the molecule based on its SMILES or atom types."""
        if mol.HasProp("smiles"):
            s = mol.GetProp("smiles")
            if s:
                return s
        # Fallback to SMILES; as a last resort, atom sequence signature
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return ",".join(str(a.GetAtomicNum()) for a in mol.GetAtoms())

    def _prepare_mol(self, mol: Chem.Mol) -> Chem.Mol:
        """Determine or restore bonds, then sanitize. Uses cached bond topology per molecule key."""
        charge = self._get_charge(mol)
        new_key = self._molecule_key(mol)

        if new_key != self._cache_key:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            self.bond_info = [
                (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in mol.GetBonds()
            ]
            self._cache_key = new_key
        else:
            if self.bond_info:
                for begin, end, bond_type in self.bond_info:
                    mol.AddBond(begin, end, bond_type)

        Chem.SanitizeMol(mol)
        return mol

    def _build_forcefield(self, mol: Chem.Mol) -> Any:
        """Build the MMFF force field for the given molecule."""
        mp = AllChem.MMFFGetMoleculeProperties(mol, **self.mol_prop)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, **self.mol_ff)
        return ff

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces for a batch of conformers using MMFF94."""
        self._ensure_single_conformer(batch)
        mol = self._to_mol(batch)
        mol = self._prepare_mol(mol)

        ff = self._build_forcefield(mol)
        energy = torch.tensor([ff.CalcEnergy()])
        grad = (
            torch.Tensor(ff.CalcGrad()) * -KCAL_PER_MOL_TO_EV
        )  # Convert to eV/Å for convergence criteria
        return energy, grad.reshape(-1, 3)

    def get_energies(self, batch: Data | Batch) -> torch.Tensor:
        """Compute energies for a batch of conformers using MMFF94."""
        self._ensure_single_conformer(batch)
        mol = self._to_mol(batch)
        mol = self._prepare_mol(mol)

        ff = self._build_forcefield(mol)
        return torch.tensor([ff.CalcEnergy()])
