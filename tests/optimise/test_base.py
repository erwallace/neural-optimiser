import pytest
import torch
from neural_optimiser.conformers import Conformer, ConformerBatch


def test_requires_calculator_set(batch, dummy_optimiser_cls):
    """Test that running without setting a calculator raises an error."""
    opt = dummy_optimiser_cls(steps=1)
    with pytest.raises(AttributeError, match="calculator must be set"):
        opt.run(batch)


def test_exit_before_any_step_when_already_converged(batch, dummy_optimiser_cls, zero_calculator):
    """Test that the optimiser exits before any step if already converged."""
    opt = dummy_optimiser_cls(steps=10, fmax=0.1)  # fmax allows early convergence
    opt.calculator = zero_calculator  # zero forces -> already converged

    converged = opt.run(batch)
    assert converged is True
    assert opt.nsteps == 0  # exited before any step
    assert batch.pos_dt.shape == (1, batch.n_atoms, 3)  # only initial frame
    assert torch.equal(
        batch.converged, torch.ones(batch.n_conformers, dtype=torch.bool, device=batch.pos.device)
    )
    assert torch.equal(
        batch.converged_step,
        torch.full((batch.n_conformers,), -1, dtype=torch.long, device=batch.pos.device),
    )
    assert hasattr(batch, "pos") and tuple(batch.pos.shape) == (batch.n_atoms, 3)
    assert hasattr(batch, "forces") and tuple(batch.forces.shape) == (batch.n_atoms, 3)
    assert hasattr(batch, "energies") and tuple(batch.energies.shape) == (batch.n_conformers,)


def test_fexit_triggers_early_exit_before_step(
    atoms, atoms2, dummy_optimiser_cls, const_calculator_factory
):
    """Test that fexit criterion triggers early exit before any step."""
    batch = ConformerBatch.from_ase([atoms, atoms2], device="cpu")

    # sqrt(3)*1.0 ≈ 1.732 > fexit -> immediate early-exit
    opt = dummy_optimiser_cls(steps=100, fmax=None, fexit=0.5)
    opt.calculator = const_calculator_factory(1.0)

    converged = opt.run(batch)
    assert converged is False
    assert opt.nsteps == 0
    assert torch.equal(
        batch.converged, torch.zeros(batch.n_conformers, dtype=torch.bool, device=batch.pos.device)
    )


def test_step_limit_and_trajectory(batch, dummy_optimiser_cls, zero_calculator):
    """Test that the optimiser respects the step limit and records the trajectory."""
    opt = dummy_optimiser_cls(steps=3, fmax=None, max_step=0.04)
    opt.calculator = zero_calculator  # no movement, but loop advances

    converged = opt.run(batch)
    assert converged is False  # exited due to step cap
    assert opt.nsteps == 3
    assert batch.pos_dt.shape == (4, batch.n_atoms, 3)  # T = steps + 1
    assert torch.allclose(batch.pos_dt[0], batch.pos_dt[-1])


def test_forces_shape_validation_raises(batch, dummy_optimiser_cls, bad_shape_calculator):
    """Test that a calculator returning forces of incorrect shape raises an error."""
    opt = dummy_optimiser_cls(steps=1, fmax=None)
    opt.calculator = bad_shape_calculator

    with pytest.raises(RuntimeError):
        opt.run(batch)


def test_single_data_synthesizes_ptr(atoms, dummy_optimiser_cls, zero_calculator):
    """Test that a single Conformer (not ConformerBatch) is handled correctly."""
    conf = Conformer.from_ase(atoms, device="cpu")

    opt = dummy_optimiser_cls(steps=2, fmax=None)
    opt.calculator = zero_calculator

    converged = opt.run(conf)
    assert converged is False
    assert opt.nsteps == 2
    assert hasattr(conf, "pos_dt") and conf.pos_dt.shape == (3, conf.pos.shape[0], 3)
