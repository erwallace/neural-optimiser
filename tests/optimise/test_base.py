from unittest.mock import MagicMock

import pytest
import torch
from neural_optimiser.conformers import Conformer, ConformerBatch
from neural_optimiser.optimisers.base import Optimiser, Trajectory


class MockOptimiser(Optimiser):
    """A concrete implementation of the abstract Optimiser for testing purposes."""

    def step(self, forces: torch.Tensor) -> None:
        """A mock step method that does nothing."""
        pass


@pytest.fixture
def mock_optimiser():
    """Fixture to create a mock optimiser instance for testing."""
    optimiser = MockOptimiser(steps=10, fmax=0.5, fexit=10.0)
    optimiser.n_confs = 2
    optimiser.batch = MagicMock()
    optimiser.batch.converged = torch.tensor([False, False])
    optimiser.nsteps = 0
    return optimiser


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
        batch.converged,
        torch.ones(batch.n_conformers, dtype=torch.bool, device=batch.pos.device),
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
        batch.converged,
        torch.zeros(batch.n_conformers, dtype=torch.bool, device=batch.pos.device),
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


def test_single_data_synthesizes_ptr(atoms, dummy_optimiser_cls, zero_calculator):
    """Test that a single Conformer (not ConformerBatch) is handled correctly."""
    conf = Conformer.from_ase(atoms, device="cpu")

    opt = dummy_optimiser_cls(steps=2, fmax=None)
    opt.calculator = zero_calculator

    converged = opt.run(conf)
    assert converged is False
    assert opt.nsteps == 2
    assert hasattr(conf, "pos_dt") and conf.pos_dt.shape == (3, conf.pos.shape[0], 3)


def test_should_exit_all_converged(mock_optimiser):
    """
    Tests that `_should_exit` returns True when all conformers have converged.
    """
    # GIVEN: All conformers have converged
    mock_optimiser.batch.converged = torch.tensor([True, True])
    fmax_per_conf = torch.tensor([0.1, 0.2])

    # WHEN: The exit condition is checked
    result = mock_optimiser._should_exit(fmax_per_conf, after_step=True)

    # THEN: The optimiser should exit and report convergence
    assert result is True
    assert mock_optimiser._converged is True


def test_should_exit_step_limit_reached(mock_optimiser):
    """
    Tests that `_should_exit` returns True when the step limit is reached.
    """
    # GIVEN: The step limit has been reached
    mock_optimiser.nsteps = 10
    fmax_per_conf = torch.tensor([0.6, 0.7])  # Not converged

    # WHEN: The exit condition is checked
    result = mock_optimiser._should_exit(fmax_per_conf, after_step=True)

    # THEN: The optimiser should exit, but not report convergence
    assert result is True
    assert mock_optimiser._converged is False


def test_should_exit_fexit_exceeded(mock_optimiser):
    """
    Tests that `_should_exit` returns True when forces on all non-converged
    conformers exceed the fexit threshold.
    """
    # GIVEN: Forces on all non-converged conformers are above fexit
    fmax_per_conf = torch.tensor([11.0, 12.0])  # Above fexit=10.0

    # WHEN: The exit condition is checked
    result = mock_optimiser._should_exit(fmax_per_conf, after_step=True)

    # THEN: The optimiser should exit and not report convergence
    assert result is True
    assert mock_optimiser._converged is False


def test_should_not_exit_fexit_with_one_below_threshold(mock_optimiser):
    """
    Tests that `_should_exit` returns False if at least one non-converged
    conformer is below the fexit threshold.
    """
    # GIVEN: One conformer is above fexit, but the other is below
    fmax_per_conf = torch.tensor([11.0, 9.0])  # One is below fexit=10.0

    # WHEN: The exit condition is checked
    result = mock_optimiser._should_exit(fmax_per_conf, after_step=True)

    # THEN: The optimiser should not exit
    assert result is False


def test_should_not_exit_when_no_conditions_met(mock_optimiser):
    """
    Tests that `_should_exit` returns False when no exit conditions are met.
    """
    # GIVEN: No exit conditions are met
    mock_optimiser.nsteps = 5
    fmax_per_conf = torch.tensor([0.6, 0.7])  # Not converged, but below fexit

    # WHEN: The exit condition is checked
    result = mock_optimiser._should_exit(fmax_per_conf, after_step=True)

    # THEN: The optimiser should not exit
    assert result is False


def test_update_convergence_records_step(mock_optimiser):
    """
    Tests that `_update_convergence` correctly records the step at which
    a conformer converges.
    """
    # GIVEN: One conformer is about to converge on step 5
    mock_optimiser.nsteps = 5
    mock_optimiser.batch.converged = torch.tensor([False, False])
    mock_optimiser.batch.converged_step = torch.tensor([-1, -1])
    fmax_per_conf = torch.tensor([0.4, 0.8])  # First conformer converges (fmax=0.5)

    # WHEN: Convergence is updated after a step
    mock_optimiser._update_convergence(fmax_per_conf, after_step=True)

    # THEN: The first conformer should be marked as converged at step 5
    assert torch.equal(mock_optimiser.batch.converged, torch.tensor([True, False]))
    assert torch.equal(mock_optimiser.batch.converged_step, torch.tensor([5, -1]))


def test_trajectory_single_conformer_finalise_and_clone(batch):
    # batch fixture: single conformer from H2O
    device = batch.pos.device
    nconf = 1
    natoms = batch.pos.size(0)

    traj = Trajectory(batch)

    # Initial properties
    e0 = torch.zeros(nconf, dtype=torch.float32, device=device)
    f0 = torch.arange(natoms, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 3)
    traj.add_initial_properties(e0, f0)

    # Add one frame and then mutate inputs to ensure cloning
    pos1 = batch.pos + 10.0
    e1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    f1 = f0 + 10.0
    pos1_before = pos1.clone()
    traj.add_frame(pos1, e1, f1)
    pos1 += 999.0  # mutation after add_frame should not affect stored trajectory

    # Converged at frame 1
    batch.converged_step = torch.tensor([1], dtype=torch.long, device=device)

    traj.finalise(batch)

    # Shape checks: 2 frames (initial + 1)
    assert batch.pos_dt.shape == (2, natoms, 3)
    assert batch.forces_dt.shape == (2, natoms, 3)
    assert batch.energies_dt.shape == (2, nconf)

    # Selected properties from converged step
    assert torch.allclose(batch.pos, pos1_before)
    assert torch.allclose(batch.forces, f1)
    assert torch.allclose(batch.energies, e1)


def test_trajectory_multi_conformer_mixed_convergence(atoms, atoms2):
    from neural_optimiser.conformers import ConformerBatch

    # Build a two-conformer batch from fixtures
    batch = ConformerBatch.from_ase([atoms, atoms2], device="cpu")
    device = batch.pos.device
    nconf = batch.n_conformers
    assert nconf == 2
    natoms = batch.pos.size(0)

    traj = Trajectory(batch)

    # Initial properties
    e0 = torch.tensor([0.0, 10.0], dtype=torch.float32, device=device)
    f0 = torch.zeros(natoms, 3, dtype=torch.float32, device=device)
    traj.add_initial_properties(e0, f0)

    # Frame 1
    pos1 = batch.pos + 10.0
    e1 = torch.tensor([1.0, 11.0], dtype=torch.float32, device=device)
    f1 = f0 + 10.0
    traj.add_frame(pos1, e1, f1)

    # Frame 2
    pos2 = batch.pos + 20.0
    e2 = torch.tensor([2.0, 12.0], dtype=torch.float32, device=device)
    f2 = f0 + 20.0
    traj.add_frame(pos2, e2, f2)

    # Mixed convergence: conf0 at frame 1, conf1 not converged -> last frame (-1)
    batch.converged_step = torch.tensor([1, -1], dtype=torch.long, device=device)

    traj.finalise(batch)

    # Shape checks: 3 frames (initial + 2)
    assert batch.pos_dt.shape == (3, natoms, 3)
    assert batch.forces_dt.shape == (3, natoms, 3)
    assert batch.energies_dt.shape == (3, nconf)

    mask0 = batch.batch == 0
    mask1 = batch.batch == 1

    # Conf 0 -> frame 1
    assert torch.allclose(batch.pos[mask0], pos1[mask0])
    assert torch.allclose(batch.forces[mask0], f1[mask0])

    # Conf 1 -> last frame (frame 2)
    assert torch.allclose(batch.pos[mask1], pos2[mask1])
    assert torch.allclose(batch.forces[mask1], f2[mask1])

    # Energies selected per conformer
    expected_energies = torch.tensor(
        [e1[0].item(), e2[1].item()], dtype=torch.float32, device=device
    )
    assert torch.allclose(batch.energies, expected_energies)
