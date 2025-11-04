import pytest
import torch
from ase.units import eV, kcal, mol
from neural_optimiser.calculators import MMFF94Calculator
from neural_optimiser.conformers import Conformer
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

KCAL_PER_MOL_TO_EV = kcal / mol / eV


def test_MMFF94Calculator_calculate(mol):
    """Compare energy and forces to RDKit's MMFF94 implementation."""
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    _energy = ff.CalcEnergy()
    _forces = torch.tensor(ff.CalcGrad()) * -KCAL_PER_MOL_TO_EV

    data = Conformer.from_rdkit(mol)
    calc = MMFF94Calculator()
    energy, forces = calc._calculate(data)

    assert torch.isclose(energy, torch.tensor(_energy))
    assert torch.allclose(forces.flatten(), torch.Tensor(_forces))


def test_repr_no_args():
    """Test the __repr__ method with no arguments."""
    calc = MMFF94Calculator()
    assert repr(calc) == "MMFFCalculator()"


def test_repr_with_args_only_affects_repr_not_behavior():
    """Test the __repr__ method with arguments."""
    calc = MMFF94Calculator(
        MMFFGetMoleculeProperties={"mmffVariant": "MMFF94s"},
        MMFFGetMoleculeForceField={"nonBondedThresh": 10.0},
    )
    s = repr(calc)
    assert "mmffVariant=MMFF94s" in s
    assert "nonBondedThresh=10.0" in s


def test_single_conformer_energy_and_forces_shape(batch):
    """Test that energy is scalar and forces have correct shape."""
    calc = MMFF94Calculator()
    energy, forces = calc._calculate(batch)

    # energy is a scalar tensor
    assert isinstance(energy, torch.Tensor)
    assert energy.ndim == 1

    # forces match [n_atoms, 3]
    assert isinstance(forces, torch.Tensor)
    assert forces.ndim == 2 and forces.shape[1] == 3
    assert forces.shape[0] == batch.pos.shape[0]


def test_get_energies_matches_calculate(batch):
    """Test that get_energies matches the energy from _calculate."""
    calc = MMFF94Calculator()
    e_only = calc.get_energies(batch)
    e_calc, _ = calc._calculate(batch)

    assert torch.allclose(e_only, e_calc)


def test_multi_conformer_raises(minimised_batch):
    """Test that multi-conformer batches raise ValueError."""
    calc = MMFF94Calculator()
    with pytest.raises(ValueError):
        calc._calculate(minimised_batch)
    with pytest.raises(ValueError):
        calc.get_energies(minimised_batch)


def test_bond_determination_cached(monkeypatch, batch):
    """Test that bond determination is cached based on molecule key."""
    calls = {"n": 0}
    orig = rdDetermineBonds.DetermineBonds

    def wrapper(mol, *args, **kwargs):
        calls["n"] += 1
        return orig(mol, *args, **kwargs)

    monkeypatch.setattr(rdDetermineBonds, "DetermineBonds", wrapper)

    calc = MMFF94Calculator()

    # First call should determine bonds once
    _ = calc._calculate(batch)
    assert calls["n"] == 1

    # Second prepare on a new mol with same key should reuse cached bonds
    mol2 = Chem.RWMol(batch.to_rdkit()[0])
    _ = calc._prepare_mol(mol2)
    assert calls["n"] == 1  # unchanged


def test_smiles_property_controls_cache_key(monkeypatch, batch):
    """Test that the 'smiles' property controls the caching of bond determination."""
    calls = {"n": 0}
    orig = rdDetermineBonds.DetermineBonds

    def wrapper(mol, *args, **kwargs):
        calls["n"] += 1
        return orig(mol, *args, **kwargs)

    monkeypatch.setattr(rdDetermineBonds, "DetermineBonds", wrapper)

    calc = MMFF94Calculator()

    # First molecule with a specific 'smiles' property
    mol1 = Chem.RWMol(batch.to_rdkit()[0])
    mol1.SetProp("smiles", "KEY")
    _ = calc._prepare_mol(mol1)
    assert calls["n"] == 1

    # Same key => should not re-run DetermineBonds
    mol2 = Chem.RWMol(batch.to_rdkit()[0])
    mol2.SetProp("smiles", "KEY")
    _ = calc._prepare_mol(mol2)
    assert calls["n"] == 1

    # Different key => should run DetermineBonds again
    mol3 = Chem.RWMol(batch.to_rdkit()[0])
    mol3.SetProp("smiles", "KEY2")
    _ = calc._prepare_mol(mol3)
    assert calls["n"] == 2
