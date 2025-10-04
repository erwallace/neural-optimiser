import pytest
import torch
from ase.build import molecule
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture(autouse=True)
def set_torch_seed():
    torch.manual_seed(42)


@pytest.fixture
def mol():
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    code = AllChem.EmbedMolecule(mol)
    if code != 0:
        pytest.skip("RDKit 3D embedding failed in this environment")
    return mol


@pytest.fixture
def mol2():
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    code = AllChem.EmbedMolecule(mol)
    if code != 0:
        pytest.skip("RDKit 3D embedding failed in this environment")
    return mol


@pytest.fixture
def atoms():
    return molecule("H2O")


@pytest.fixture
def atoms2():
    return molecule("NH3")
