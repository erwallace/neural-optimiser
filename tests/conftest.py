import pytest
import torch
from ase.build import molecule
from neural_optimiser import test_dir
from neural_optimiser.calculators import MACECalculator
from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.optimisers.base import Optimiser
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture(autouse=True)
def set_torch_seed():
    """Set a fixed random seed for PyTorch for reproducible tests."""
    torch.manual_seed(42)


@pytest.fixture
def mol():
    """Simple RDKit molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    code = AllChem.EmbedMolecule(mol)
    if code != 0:
        pytest.skip("RDKit 3D embedding failed in this environment")
    return mol


@pytest.fixture
def mol2():
    """Simple RDKit molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    code = AllChem.EmbedMolecule(mol)
    if code != 0:
        pytest.skip("RDKit 3D embedding failed in this environment")
    return mol


@pytest.fixture
def atoms():
    """Simple ASE Atoms object."""
    return molecule("H2O")


@pytest.fixture
def atoms2():
    """Simple ASE Atoms object."""
    return molecule("NH3")


@pytest.fixture
def batch(atoms):
    """Single-conformer batch for testing."""
    return ConformerBatch.from_ase([atoms], device="cpu")


@pytest.fixture
def minimised_batch(atoms, atoms2):
    """Two-conformer batch with optimisation trajectory data for testing."""
    batch = ConformerBatch.from_ase([atoms, atoms2], device="cpu")

    # Attach optimisation trajectory tensors: 2 steps
    pos = batch.pos
    batch.forces = torch.zeros_like(pos)  # [n_atoms, 3]

    batch.pos_dt = torch.stack([pos + 0.1, pos + 0.2], dim=0)  # [2, n_atoms, 3]
    batch.forces_dt = torch.stack(
        [batch.forces.clone(), batch.forces.clone()], dim=0
    )  # [2, n_atoms, 3]

    # Per-conformer energies and per-step energies
    batch.energies = torch.tensor([-1.0, -2.0], dtype=torch.float32)  # [n_confs]
    batch.energies_dt = torch.tensor(  # [2, n_confs]
        [[-0.8, -1.8], [-1.0, -2.0]], dtype=torch.float32
    )

    return batch


class DummyOptimiser(Optimiser):
    """Minimal concrete optimiser; performs capped steepest-descent steps."""

    def step(self, forces: torch.Tensor) -> None:
        for _, (s, e) in self._iter_conformer_slices():
            F = forces[s:e]
            norms = F.norm(dim=1, keepdim=True).clamp(min=1e-12)
            step = F / norms * torch.clamp(norms, max=self.max_step)
            self.batch.pos[s:e] = self.batch.pos[s:e] + step


class ZeroCalculator:
    """Calculator that returns zero energy and forces."""

    def __call__(self, batch):
        return torch.zeros(batch.n_conformers), torch.zeros_like(batch.pos)


class ConstCalculator:
    """Constant forces and zero energy: value applied to all atoms."""

    def __init__(self, value: float):
        self.value = float(value)

    def __call__(self, batch):
        return torch.zeros(batch.n_conformers), torch.full_like(batch.pos, self.value)


class PerConfConstCalculator:
    """Constant forces per conformer: values[i] applied to all atoms in conformer i."""

    def __init__(self, values):
        self.values = [float(v) for v in values]

    def __call__(self, batch):
        f = torch.zeros_like(batch.pos)
        for i, v in enumerate(self.values):
            mask = batch.batch == i
            f[mask] = v  # broadcasts to 3 components
        return torch.zeros(batch.n_conformers), f


@pytest.fixture
def dummy_optimiser_cls():
    """A minimal concrete optimiser class."""
    return DummyOptimiser


@pytest.fixture
def zero_calculator():
    """A calculator that returns zero energy and forces."""
    return ZeroCalculator()


@pytest.fixture
def const_calculator_factory():
    """A factory for calculators that return constant forces."""

    def _make(value: float):
        return ConstCalculator(value)

    return _make


@pytest.fixture
def per_conf_const_calculator_factory():
    """A factory for calculators that return constant forces per conformer."""

    def _make(values):
        return PerConfConstCalculator(values)

    return _make


@pytest.fixture
def mace_calculator(scope="session"):
    """A MACE calculator for testing (if MACE is installed)."""
    pytest.importorskip("mace", reason="MACE not installed")

    model_paths = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"
    return MACECalculator(model_paths=str(model_paths), device="cpu")
