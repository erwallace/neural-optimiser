import pytest
import torch


def test_calculate_returns_expected_shapes(fairchem_calculator, batch):
    """Test that calculate returns tensors of expected shapes and dtypes."""
    e, f = fairchem_calculator(batch)

    assert isinstance(e, torch.Tensor)
    assert isinstance(f, torch.Tensor)
    assert e.shape == (batch.n_conformers,)
    assert f.shape == batch.pos.shape
    assert f.dtype == batch.pos.dtype


def test_fairchem_calculator(fairchem_calculator, fairchem_model_path, batch, atoms, device):
    """Compare FAIRChemCalculator results to ASE FAIRChem calculator."""
    pytest.importorskip("fairchem", reason="FAIRChem not installed")
    from fairchem.core import FAIRChemCalculator as FAIRChemCalc
    from fairchem.core.units.mlip_unit import load_predict_unit

    e, f = fairchem_calculator(batch)

    predictor = load_predict_unit(path=fairchem_model_path, device=device)
    fairchem_calc = FAIRChemCalc(predictor)
    atoms.calc = fairchem_calc
    _e = torch.tensor(atoms.get_potential_energy(), dtype=e.dtype, device=device)
    _f = torch.tensor(atoms.get_forces(), dtype=f.dtype, device=device)

    # Ensure comparable shapes/dtypes
    print(f)
    print(_f)
    print(f - _f)

    assert torch.isclose(e.squeeze(), _e, atol=1e-4)
    assert torch.allclose(f, _f, atol=1e-4)


def test_fairchem_calculator2(fairchem_calculator, fairchem_model_path, batch, atoms, device):
    """Compare FAIRChemCalculator results to ASE FAIRChem calculator."""
    pytest.importorskip("fairchem", reason="FAIRChem not installed")
    from fairchem.core import FAIRChemCalculator as FAIRChemCalc
    from fairchem.core.units.mlip_unit import load_predict_unit

    e = fairchem_calculator.get_energies(batch)

    predictor = load_predict_unit(path=fairchem_model_path, device=device)
    fairchem_calc = FAIRChemCalc(predictor)
    atoms.calc = fairchem_calc
    _e = atoms.get_potential_energy()

    # Ensure comparable shapes/dtypes
    assert torch.isclose(e.squeeze(), torch.tensor(_e, dtype=e.dtype).to(device), atol=1e-4)
