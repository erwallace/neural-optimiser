import pytest
import torch
from ase.build import molecule
from neural_optimiser import test_dir
from neural_optimiser.calculators import MACECalculator
from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.optimise.base import Optimiser
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


@pytest.fixture
def batch(atoms):
    return ConformerBatch.from_ase([atoms], device="cpu")


class DummyOptimiser(Optimiser):
    """Minimal concrete optimiser; performs capped steepest-descent steps."""

    def step(self, forces: torch.Tensor) -> None:
        for _, (s, e) in self._iter_conformer_slices():
            F = forces[s:e]
            norms = F.norm(dim=1, keepdim=True).clamp(min=1e-12)
            step = F / norms * torch.clamp(norms, max=self.max_step)
            self.batch.pos[s:e] = self.batch.pos[s:e] + step


class ZeroCalculator:
    def calculate(self, batch):
        return 0.0, torch.zeros_like(batch.pos)


class ConstCalculator:
    def __init__(self, value: float):
        self.value = float(value)

    def calculate(self, batch):
        return 0.0, torch.full_like(batch.pos, self.value)


class BadShapeCalculator:
    def calculate(self, batch):
        return 0.0, torch.zeros(batch.pos.shape[0], device=batch.pos.device, dtype=batch.pos.dtype)


class PerConfConstCalculator:
    """Constant forces per conformer: values[i] applied to all atoms in conformer i."""

    def __init__(self, values):
        self.values = [float(v) for v in values]

    def calculate(self, batch):
        f = torch.zeros_like(batch.pos)
        for i, v in enumerate(self.values):
            mask = batch.batch == i
            f[mask] = v  # broadcasts to 3 components
        return 0.0, f


@pytest.fixture
def dummy_optimiser_cls():
    return DummyOptimiser


@pytest.fixture
def zero_calculator():
    return ZeroCalculator()


@pytest.fixture
def const_calculator_factory():
    def _make(value: float):
        return ConstCalculator(value)

    return _make


@pytest.fixture
def bad_shape_calculator():
    return BadShapeCalculator()


@pytest.fixture
def per_conf_const_calculator_factory():
    def _make(values):
        return PerConfConstCalculator(values)

    return _make


@pytest.fixture
def mace_calculator(scope="session"):
    pytest.importorskip("mace", reason="MACE not installed")

    model_paths = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"
    return MACECalculator(model_paths=str(model_paths), device="cpu")
