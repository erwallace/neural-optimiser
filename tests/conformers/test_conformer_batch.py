import numpy as np
import torch
from neural_optimiser.conformers import Conformer, ConformerBatch
from rdkit import Chem


def test_from_ase_batch_properties(atoms, atoms2):
    batch = ConformerBatch.from_ase([atoms, atoms2])

    # Dtypes
    assert batch.atom_types.dtype == ConformerBatch.atom_type_dtype
    assert batch.pos.dtype == ConformerBatch.pos_dtype

    # Sizes
    assert batch.n_conformers == 2
    assert batch.n_atoms == len(atoms) + len(atoms2)

    print(batch)

    # molecule_idxs should be created and equal to batch mapping
    assert hasattr(batch, "molecule_idxs")
    assert torch.equal(batch.molecule_idxs, batch.batch)
    assert batch.n_molecules == batch.n_conformers  # by construction for from_ase()

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


def test_from_rdkit_batch_properties(mol, mol2):
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
    assert batch.molecule_idxs.dtype == ConformerBatch.molecule_idx_dtype

    # molecule_idxs should map each conformer chunk to the correct molecule index
    expected_mol_for_conf = [0, 0, 1]
    for conf_idx, expected_mid in enumerate(expected_mol_for_conf):
        mask = batch.batch == conf_idx
        labels = batch.molecule_idxs[mask]
        uniq = torch.unique(labels)
        assert uniq.numel() == 1
        assert int(uniq.item()) == expected_mid

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
    batch = ConformerBatch.from_rdkit(mol)
    assert batch.n_molecules == 1
    assert torch.unique(batch.molecule_idxs).tolist() == [0]
    assert batch.n_conformers == mol.GetNumConformers()
