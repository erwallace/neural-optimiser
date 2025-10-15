import numpy as np
import torch
from neural_optimiser.conformers import Conformer, ConformerBatch
from rdkit import Chem


def test_from_ase(atoms, atoms2):
    """Test creating a ConformerBatch from multiple ASE Atoms objects,"""
    batch = ConformerBatch.from_ase([atoms, atoms2])

    # Dtypes
    assert batch.atom_types.dtype == ConformerBatch.atom_type_dtype
    assert batch.pos.dtype == ConformerBatch.pos_dtype

    # Sizes
    assert batch.n_conformers == 2
    assert batch.n_atoms == len(atoms) + len(atoms2)

    # Slicing back each conformer matches originals
    conf0 = batch.conformer(0)
    conf1 = batch.conformer(1)

    assert isinstance(conf0, Conformer) and isinstance(conf1, Conformer)
    assert conf0.atom_types.dtype == Conformer.atom_type_dtype
    assert conf0.pos.dtype == Conformer.pos_dtype
    assert conf1.atom_types.dtype == Conformer.atom_type_dtype
    assert conf1.pos.dtype == Conformer.pos_dtype

    atoms0 = conf0.to_ase()
    atoms1 = conf1.to_ase()
    np.testing.assert_array_equal(atoms.get_atomic_numbers(), atoms0.get_atomic_numbers())
    np.testing.assert_allclose(atoms.get_positions(), atoms0.get_positions(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(atoms2.get_atomic_numbers(), atoms1.get_atomic_numbers())
    np.testing.assert_allclose(atoms2.get_positions(), atoms1.get_positions(), rtol=0, atol=1e-6)


def test_from_rdkit_batch(mol, mol2):
    """Test creating a ConformerBatch from multiple RDKit Mol objects."""
    # Add a second conformer to the first molecule
    new_conf = Chem.Conformer(mol.GetConformer(0))
    mol.AddConformer(new_conf)

    batch = ConformerBatch.from_rdkit([mol, mol2])

    # Expected counts
    n_atoms = 2 * mol.GetNumAtoms() + 1 * mol2.GetNumAtoms()
    assert batch.n_molecules == 2
    assert batch.n_conformers == 3
    assert batch.n_atoms == n_atoms

    # Dtypes
    assert batch.atom_types.dtype == ConformerBatch.atom_type_dtype
    assert batch.pos.dtype == ConformerBatch.pos_dtype

    # Slicing back conformers has correct sizes and types
    c0 = batch.conformer(0)
    c1 = batch.conformer(1)
    c2 = batch.conformer(2)

    assert c0.atom_types.shape == (mol.GetNumAtoms(),)
    assert c1.atom_types.shape == (mol.GetNumAtoms(),)
    assert c2.atom_types.shape == (mol2.GetNumAtoms(),)
    assert c0.pos.shape == (mol.GetNumAtoms(), 3)
    assert c1.pos.shape == (mol.GetNumAtoms(), 3)
    assert c2.pos.shape == (mol2.GetNumAtoms(), 3)

    assert c0.atom_types.dtype == Conformer.atom_type_dtype
    assert c0.pos.dtype == Conformer.pos_dtype
    assert c2.atom_types.dtype == Conformer.atom_type_dtype
    assert c2.pos.dtype == Conformer.pos_dtype


def test_from_rdkit_single_mol(mol):
    """Test creating a ConformerBatch from a single RDKit Mol object with multiple conformers."""
    batch = ConformerBatch.from_rdkit(mol)
    assert batch.n_molecules == 1
    assert batch.n_conformers == mol.GetNumConformers()


def test_conformer(minimised_batch: ConformerBatch):
    """Test slicing conformers at specific optimisation steps."""
    b = minimised_batch

    # Check conformer 1 without step (uses base pos/forces)
    c1 = b.conformer(idx=1)

    for attr in b.__dict__["_store"]:
        if attr not in ["batch", "ptr", "device"]:
            assert hasattr(c1, attr)

    mask1 = b.batch == 1
    assert torch.allclose(c1.pos, b.pos[mask1], atol=1e-7)
    assert torch.allclose(c1.forces, b.forces[mask1], atol=1e-7)


def test_conformer_with_step(minimised_batch: ConformerBatch):
    """Test slicing conformers at specific optimisation steps."""
    b = minimised_batch
    # Check conformer 0 at step 1
    c0 = b.conformer(idx=0, step=1)
    mask0 = b.batch == 0

    assert c0.pos.shape == (mask0.sum().item(), 3)
    assert torch.allclose(c0.pos, b.pos_dt[1][mask0], atol=1e-7)
    assert torch.allclose(c0.forces, b.forces_dt[1][mask0], atol=1e-7)
    assert torch.isclose(c0.energies, b.energies_dt[1, 0])


def test_to_data_list(minimised_batch: ConformerBatch, atoms, atoms2):
    """Test converting a ConformerBatch to a list of Data objects."""
    b = minimised_batch
    data_list = b.to_data_list()

    assert isinstance(data_list, list)
    assert len(data_list) == 2

    # H2O has 3 atoms, NH3 has 4
    expected_counts = [len(atoms), len(atoms2)]
    for i, d in enumerate(data_list):
        assert hasattr(d, "pos")
        assert d.pos.shape == (expected_counts[i], 3)


def test_to_rdkit(minimised_batch: ConformerBatch, atoms, atoms2):
    """Test converting a ConformerBatch to a list of RDKit Mol objects."""
    mols = minimised_batch.to_rdkit()

    assert isinstance(mols, list)
    assert len(mols) == 2
    assert mols[0].GetNumAtoms() == len(atoms)
    assert mols[1].GetNumAtoms() == len(atoms2)


def test_to_ase(minimised_batch: ConformerBatch, atoms, atoms2):
    """Test converting a ConformerBatch to a list of ASE Atoms objects."""
    atoms_list = minimised_batch.to_ase()

    assert isinstance(atoms_list, list)
    assert len(atoms_list) == 2
    assert len(atoms_list[0]) == len(atoms)
    assert len(atoms_list[1]) == len(atoms2)


def test_cat(atoms, atoms2):
    """Test concatenating multiple ConformerBatch objects."""
    b1 = ConformerBatch.from_ase([atoms], device="cpu")
    b2 = ConformerBatch.from_ase([atoms2], device="cpu")
    cat = ConformerBatch.cat([b1, b2])
    assert isinstance(cat, ConformerBatch)
    assert cat.n_conformers == b1.n_conformers + b2.n_conformers
    assert cat.n_atoms == len(atoms) + len(atoms2)
    # Check per-conformer atom counts preserved in order
    counts = [d.pos.shape[0] for d in cat.to_data_list()]
    assert counts == [len(atoms), len(atoms2)]


def test_from_data_to_data_roundtrip(minimised_batch: ConformerBatch):
    """Test converting a ConformerBatch to a list of Data objects and back."""
    data_list = minimised_batch.to_data_list()
    new_batch = ConformerBatch.from_data_list(data_list, device="cpu")
    assert isinstance(new_batch, ConformerBatch)
    assert new_batch.n_conformers == minimised_batch.n_conformers
    assert new_batch.n_atoms == minimised_batch.n_atoms
    # Per-conformer atom counts preserved
    orig_counts = [d.pos.shape[0] for d in data_list]
    new_counts = [d.pos.shape[0] for d in new_batch.to_data_list()]
    assert new_counts == orig_counts


def test_from_rdkit_to_rdkit_roundtrip(mol, mol2):
    """Test converting a ConformerBatch to a list of RDKit Mol objects and back."""
    # Add a second conformer to the first molecule
    mol.AddConformer(Chem.Conformer(mol.GetConformer(0)))
    batch = ConformerBatch.from_rdkit([mol, mol2])

    new_mols = batch.to_rdkit()
    assert isinstance(new_mols, list)
    assert len(new_mols) == 3  # conformers of the same molecule are separate
    assert new_mols[0].GetNumAtoms() == mol.GetNumAtoms()
    assert new_mols[-1].GetNumAtoms() == mol2.GetNumAtoms()


def test_from_ase_to_ase_roundtrip(atoms, atoms2):
    """Test converting a ConformerBatch to a list of ASE Atoms objects and back."""
    batch = ConformerBatch.from_ase([atoms, atoms2])
    new_atoms_list = batch.to_ase()
    assert isinstance(new_atoms_list, list)
    assert len(new_atoms_list) == 2
    assert len(new_atoms_list[0]) == len(atoms)
    assert len(new_atoms_list[1]) == len(atoms2)
