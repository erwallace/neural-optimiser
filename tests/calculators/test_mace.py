import pytest
import torch
from ase.build import molecule
from neural_optimiser import test_dir
from neural_optimiser.calculators._mace import MACECalculator
from neural_optimiser.conformers import ConformerBatch


def test_mace_calculator():
    pytest.importorskip("mace", reason="MACE not installed")
    from mace.calculators.mace import MACECalculator as MACECalc

    model_paths = test_dir / "models" / "MACE_SPICE2_NEUTRAL.model"

    atoms = molecule("H2O")
    batch = ConformerBatch.from_ase([atoms], device="cpu")

    calculator = MACECalculator(model_paths=str(model_paths), device="cpu")
    e, f = calculator.calculate(batch)

    mace_calc = MACECalc(model_paths=str(model_paths), device="cpu")
    atoms.calc = mace_calc
    _e = atoms.get_potential_energy()
    _f = atoms.get_forces()

    # Ensure comparable shapes/dtypes
    assert torch.isclose(e.squeeze(), torch.tensor(_e, dtype=e.dtype), atol=1e-4)
    assert torch.allclose(f, torch.tensor(_f, dtype=f.dtype), atol=1e-4)
