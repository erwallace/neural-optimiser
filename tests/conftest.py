import pytest
import torch
from ase import Atoms
from ase.build import molecule
from neural_optimiser import test_dir
from neural_optimiser.calculators import FAIRChemCalculator, MACECalculator
from neural_optimiser.calculators.base import Calculator
from neural_optimiser.conformers import ConformerBatch
from neural_optimiser.optimisers.base import Optimiser
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture(scope="session")
def device() -> str:
    """Device for testing: 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def set_torch_seed():
    """Set a fixed random seed for PyTorch for reproducible tests."""
    torch.manual_seed(42)


@pytest.fixture
def mol() -> Chem.Mol:
    """Simple RDKit molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    code = AllChem.EmbedMolecule(mol)
    if code != 0:
        pytest.skip("RDKit 3D embedding failed in this environment")
    return mol


@pytest.fixture
def mol2() -> Chem.Mol:
    """Simple RDKit molecule with 3D coordinates."""
    mol = Chem.MolFromSmiles("CC")
    mol = Chem.AddHs(mol)
    code = AllChem.EmbedMolecule(mol)
    if code != 0:
        pytest.skip("RDKit 3D embedding failed in this environment")
    return mol


@pytest.fixture
def atoms() -> Atoms:
    """Simple ASE Atoms object."""
    return molecule("H2O")


@pytest.fixture
def atoms2() -> Atoms:
    """Simple ASE Atoms object."""
    return molecule("NH3")


@pytest.fixture
def batch(atoms, device) -> ConformerBatch:
    """Single-conformer batch for testing."""
    batch = ConformerBatch.from_ase([atoms])
    batch = batch.to(device)
    return batch


@pytest.fixture
def minimised_batch(atoms, atoms2, device) -> ConformerBatch:
    """Two-conformer batch with optimisation trajectory data for testing."""
    batch = ConformerBatch.from_ase([atoms, atoms2])

    # Attach optimisation trajectory tensors: 4 steps
    pos = batch.pos
    batch.forces = torch.zeros_like(pos)  # [n_atoms, 3]

    batch.pos_dt = torch.stack(
        [pos + 0.1, pos + 0.2, pos + 0.3, pos + 0.4], dim=0
    )  # [4, n_atoms, 3]
    batch.forces_dt = torch.stack(
        [batch.forces.clone(), batch.forces.clone(), batch.forces.clone(), batch.forces.clone()],
        dim=0,
    )  # [4, n_atoms, 3]

    # Per-conformer energies and per-step energies
    batch.energies = torch.tensor([-1.0, -2.0], dtype=torch.float32)  # [n_confs]
    batch.energies_dt = torch.tensor(  # [4, n_confs]
        [[-0.8, -1.8], [-1.0, -2.0], [-1.2, -2.2], [-1.4, -2.4]], dtype=torch.float32
    )

    batch = batch.to(device)

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
def dummy_optimiser_cls() -> Optimiser:
    """A minimal concrete optimiser class."""
    return DummyOptimiser


@pytest.fixture
def zero_calculator() -> Calculator:
    """A calculator that returns zero energy and forces."""
    return ZeroCalculator()


@pytest.fixture
def const_calculator_factory() -> Calculator:
    """A factory for calculators that return constant forces."""

    def _make(value: float):
        return ConstCalculator(value)

    return _make


@pytest.fixture
def per_conf_const_calculator_factory() -> Calculator:
    """A factory for calculators that return constant forces per conformer."""

    def _make(values):
        return PerConfConstCalculator(values)

    return _make


@pytest.fixture(scope="session")
def mace_calculator(device: str) -> MACECalculator:
    """A MACE calculator for testing (if MACE is installed)."""
    pytest.importorskip("mace", reason="MACE not installed")

    model_paths = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"
    return MACECalculator(model_paths=str(model_paths), device=device)


@pytest.fixture(scope="session")
def fairchem_model_path() -> str:
    """Path to FAIRChem test model."""
    path = test_dir / "models" / "omol25_esen_sm_direct.pt"
    if not path.exists():
        pytest.skip(f"FAIRChem test model not found ({path}).")
    return path


@pytest.fixture(scope="session")
def fairchem_calculator(device: str, fairchem_model_path: str) -> FAIRChemCalculator:
    """A FAIRChem calculator for testing (if FAIRChem is installed)."""
    pytest.importorskip("fairchem", reason="FAIRChem not installed")
    return FAIRChemCalculator(model_paths=str(fairchem_model_path), device=device)
