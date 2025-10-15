import torch
from neural_optimiser.calculators import RandomCalculator


def test_random_calculator_shapes_and_dtypes(batch):
    """Test that RandomCalculator returns tensors of expected shapes and dtypes."""
    calc = RandomCalculator()
    e, f = calc(batch)

    assert isinstance(e, torch.Tensor)
    assert isinstance(f, torch.Tensor)
    assert e.shape == (batch.n_conformers,)
    assert f.shape == batch.pos.shape
    assert f.dtype == batch.pos.dtype


def test_get_energies_shape(batch):
    """RandomCalculator.get_energies returns a 1D tensor of length n_conformers."""
    calc = RandomCalculator()
    e = calc.get_energies(batch)

    assert isinstance(e, torch.Tensor)
    assert e.shape == (batch.n_conformers,)


def test_random_calculator_reproducible_with_manual_seed(batch):
    """Test that RandomCalculator produces reproducible results with the same manual seed."""
    torch.manual_seed(0)
    calc1 = RandomCalculator()
    e1, f1 = calc1(batch)

    torch.manual_seed(0)
    calc2 = RandomCalculator()
    e2, f2 = calc2(batch)

    assert torch.allclose(e1, e2)
    assert torch.allclose(f1, f2)
