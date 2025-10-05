import numpy as np
import pytest
import torch
from neural_optimiser.conformers import Conformer


def test_from_ase_and_to_ase_roundtrip(atoms):
    """Test ASE -> Conformer -> ASE roundtrip."""
    conf = Conformer.from_ase(atoms)

    assert conf.atom_types.dtype == Conformer.atom_type_dtype
    assert conf.pos.dtype == Conformer.pos_dtype

    assert conf.atom_types.shape == (len(atoms),)
    assert conf.pos.shape == (len(atoms), 3)

    atoms2 = conf.to_ase()
    np.testing.assert_array_equal(atoms.get_atomic_numbers(), atoms2.get_atomic_numbers())
    np.testing.assert_allclose(atoms.get_positions(), atoms2.get_positions(), rtol=0, atol=1e-6)


def test_from_rdkit_and_to_rdkit_roundtrip(mol):
    """Test RDKit -> Conformer -> RDKit roundtrip."""
    conf = Conformer.from_rdkit(mol)

    n = mol.GetNumAtoms()
    assert conf.atom_types.dtype == Conformer.atom_type_dtype
    assert conf.pos.dtype == Conformer.pos_dtype
    assert conf.atom_types.shape == (n,)
    assert conf.pos.shape == (n, 3)
    assert isinstance(conf.smiles, str) and len(conf.smiles) > 0

    mol2 = conf.to_rdkit()
    assert mol2.GetNumAtoms() == n
    assert mol2.GetNumConformers() == 1

    # Positions roundtrip
    rdconf = mol2.GetConformer()
    pos2 = np.array(
        [
            [rdconf.GetAtomPosition(i).x, rdconf.GetAtomPosition(i).y, rdconf.GetAtomPosition(i).z]
            for i in range(n)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(conf.pos.detach().cpu().numpy(), pos2, rtol=0, atol=1e-5)

    # Smiles stored as property if available
    assert mol2.HasProp("_Smiles")
    assert mol2.GetProp("_Smiles") == conf.smiles


def test_validation_shape_mismatch_raises():
    """Test that shape mismatches raise ValueError."""
    atom_types = torch.tensor([1, 8], dtype=Conformer.atom_type_dtype)
    pos = torch.zeros((3, 3), dtype=Conformer.pos_dtype)
    with pytest.raises(ValueError, match="atom_types and pos must have matching n_atoms"):
        Conformer(atom_types=atom_types, pos=pos)


def test_validation_wrong_atom_types_dtype_raises():
    """Test that wrong dtype for atom_types raises ValueError."""
    atom_types = torch.tensor([1, 6, 8], dtype=torch.int32)  # wrong dtype
    pos = torch.zeros((3, 3), dtype=Conformer.pos_dtype)
    with pytest.raises(ValueError, match="atom_types must have dtype"):
        Conformer(atom_types=atom_types, pos=pos)


def test_validation_wrong_pos_dtype_raises():
    """Test that wrong dtype for pos raises ValueError."""
    atom_types = torch.tensor([1, 6, 8], dtype=Conformer.atom_type_dtype)
    pos = torch.zeros((3, 3), dtype=torch.float64)  # wrong dtype
    with pytest.raises(ValueError, match="pos must have dtype"):
        Conformer(atom_types=atom_types, pos=pos)


def test_validation_wrong_pos_shape_raises():
    """Test that wrong shape for pos raises ValueError."""
    atom_types = torch.tensor([1, 6, 8], dtype=Conformer.atom_type_dtype)
    pos = torch.zeros((3, 2), dtype=Conformer.pos_dtype)  # not (n, 3)
    with pytest.raises(ValueError, match="pos must have shape"):
        Conformer(atom_types=atom_types, pos=pos)


def test_example_rdkit_plot(mol):
    """Test RDKit Conformer plotting."""
    conformer = Conformer.from_rdkit(mol)
    assert str(conformer)

    conformer.plot(dim=2)
    conformer.plot(dim=3)


def test_example_ase_plot(atoms):
    """Test ASE Conformer plotting."""
    conformer_ase = Conformer.from_ase(atoms)
    assert str(conformer_ase)

    conformer_ase.plot(dim=3)
