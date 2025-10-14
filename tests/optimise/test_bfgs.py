import math

import pytest
import torch
from neural_optimiser.conformers import Conformer, ConformerBatch
from neural_optimiser.optimisers import BFGS


def _per_conf_step_norms(batch):
    disp = batch.pos_dt[-1] - batch.pos_dt[0]
    norms = torch.linalg.vector_norm(disp, dim=1)
    max_norms = []
    for i in range(batch.n_conformers):
        s = int(batch.ptr[i].item())
        e = int(batch.ptr[i + 1].item())
        max_norms.append(float(norms[s:e].max().item()) if e > s else 0.0)
    return max_norms


def test_bfgs_requires_calculator_set(atoms):
    batch = ConformerBatch.from_ase([atoms], device="cpu")
    opt = BFGS(steps=1)
    with pytest.raises(AttributeError, match="calculator must be set"):
        opt.run(batch)


def test_bfgs_initial_convergence_with_zero_forces(atoms, zero_calculator):
    batch = ConformerBatch.from_ase([atoms], device="cpu")

    opt = BFGS(steps=10, fmax=0.1)
    opt.calculator = zero_calculator

    converged = opt.run(batch)
    assert converged is True
    assert opt.nsteps == 0
    assert batch.pos_dt.shape == (1, batch.n_atoms, 3)
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


def test_bfgs_step_capped_and_state_updated(atoms, const_calculator_factory):
    batch = ConformerBatch.from_ase([atoms], device="cpu")

    opt = BFGS(steps=1, fmax=None, max_step=0.04)
    opt.calculator = const_calculator_factory(10.0)

    converged = opt.run(batch)
    assert converged is False
    assert opt.nsteps == 1
    max_norms = _per_conf_step_norms(batch)
    assert len(max_norms) == 1
    assert max_norms[0] <= 0.04 + 1e-6 and max_norms[0] >= 0.039
    assert len(opt._H) == batch.n_conformers and 0 in opt._H
    assert len(opt._r0) == batch.n_conformers and 0 in opt._r0
    assert len(opt._f0) == batch.n_conformers and 0 in opt._f0


def test_bfgs_batched_independent_scaling(atoms, atoms2, per_conf_const_calculator_factory):
    batch = ConformerBatch.from_ase([atoms, atoms2], device="cpu")
    opt = BFGS(steps=1, fmax=None, max_step=0.04)
    opt.calculator = per_conf_const_calculator_factory([1.0, 10.0])

    converged = opt.run(batch)
    assert converged is False
    assert opt.nsteps == 1

    expected0 = math.sqrt(3.0) * 1.0 / 70.0
    max_norms = _per_conf_step_norms(batch)
    assert abs(max_norms[0] - expected0) < 5e-3 and max_norms[0] < 0.04
    assert max_norms[1] <= 0.04 + 1e-6 and max_norms[1] >= 0.039
    assert set(opt._H.keys()) == {0, 1}


def test_bfgs_single_data_supported(atoms, const_calculator_factory):
    conf = Conformer.from_ase(atoms, device="cpu")

    opt = BFGS(steps=2, fmax=None)
    opt.calculator = const_calculator_factory(1.0)

    converged = opt.run(conf)
    assert converged is False
    assert opt.nsteps == 2
    assert hasattr(conf, "pos_dt") and conf.pos_dt.shape == (3, conf.pos.shape[0], 3)
