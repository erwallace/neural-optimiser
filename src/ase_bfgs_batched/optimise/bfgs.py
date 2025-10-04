# python
# filepath: /Users/wallace5/ase-bfgs-batched/src/ase_bfgs_batched/optimise/bfgs.py

import torch
from loguru import logger
from torch_geometric.data import Batch, Data


class BFGSBatched:
    """BFGS optimiser with additional exit conditions, operating on a PyG Batch/Data.

    - Maintains one Hessian per conformer in the batch.
    - Exits on convergence (fmax), explosion (fexit) or step limit.
    - Expects `self.calculator.forces(batch)` -> torch.Tensor[n_atoms, 3].
    - Tracks per-conformer convergence and converged step.
    - Stores per-step positions in `batch.pos_dt` and returns `batch.pos_min` on exit.
    """

    def __init__(
        self,
        max_step: float = 0.04,
        steps: int = -1,
        fmax: float | None = None,
        fexit: float | None = None,
    ) -> None:
        """Create a batched BFGS optimizer.

        Args:
            max_step: Maximum allowed displacement per-atom per step.
            steps: Maximum number of steps. If -1, only fmax/fexit control exit.
            fmax: Convergence threshold on max per-atom force norm.
            fexit: Early-exit threshold if any per-atom force exceeds this value.
        """
        self.fexit = fexit
        self.steps = steps
        self.max_step = max_step
        self.fmax = fmax
        self.calculator = None

        # BFGS state (per conformer)
        self._H: dict[int, torch.Tensor] = {}
        self._r0: dict[int, torch.Tensor] = {}
        self._f0: dict[int, torch.Tensor] = {}

        self.nsteps: int = 0
        self._converged: bool = False
        self._n_fmax: float = 0.0

        logger.debug(
            "Initialized BFGSBatched(max_step={}, steps={}, fmax={}, fexit={})",
            max_step,
            steps,
            fmax,
            fexit,
        )
        self.__post_init__()

    def __post_init__(self):
        """Validate init-time invariants."""
        if self.steps == -1 and self.fmax is None:
            raise ValueError("Either fmax or steps must be set to define convergence.")
        logger.debug("Post-init check passed.")

    def run(self, batch: Data | Batch) -> bool:
        """Run until exit conditions are met.

        Returns:
            True if all conformers converged, else False.
        """
        if self.calculator is None:
            raise AttributeError("BFGSBatched.calculator must be set before running dynamics.")

        self.device = batch.pos.device
        self.dtype = batch.pos.dtype
        self.n_atoms = batch.n_atoms
        self.batch = batch

        self._check_batch()

        logger.info(
            "Starting BFGS: nconf={}, natoms={}, steps={}, fmax={}, fexit={}, max_step={}",
            batch.n_conformers,
            self.n_atoms,
            self.steps,
            self.fmax,
            self.fexit,
            self.max_step,
        )
        return self._run()

    # --------------------- internal API ---------------------

    def _run(self) -> bool:
        """Internal driver loop. Computes forces, steps, and exit handling."""
        self.nsteps = 0
        self._converged = False

        # Init per-step trajectory storage directly on batch: [T, N, 3]
        self.batch.pos_dt = self.batch.pos.clone().unsqueeze(dim=0)

        # Initial force evaluation
        forces = self._forces()
        fmax_per_conf = self._per_conformer_max_force(forces)
        logger.debug(
            "Initial per-conformer fmax: min={:.6f}, max={:.6f}",
            float(fmax_per_conf.min().item()),
            float(fmax_per_conf.max().item()),
        )

        # Update per-conformer converged mask (no step index recorded yet)
        self._update_convergence(fmax_per_conf, after_step=False)

        # Exit checks before any step
        if self._should_exit(fmax_per_conf, after_step=False):
            logger.info(
                "Exiting before any step: converged={}, nsteps={}", self._converged, self.nsteps
            )
            self._finalize_trajectories()
            return self._converged

        while True:
            with torch.no_grad():
                self.step(forces)

            # Append current positions as a new frame to batch.pos_dt
            self.batch.pos_dt = torch.cat(
                (self.batch.pos_dt, self.batch.pos.detach().clone().unsqueeze(0)),
                dim=0,
            )
            self.nsteps += 1
            logger.debug("Completed step {}.", self.nsteps)

            forces = self._forces()
            fmax_per_conf = self._per_conformer_max_force(forces)
            logger.debug(
                "Step {} per-conformer fmax: min={:.6f}, max={:.6f}",
                self.nsteps,
                float(fmax_per_conf.min().item()),
                float(fmax_per_conf.max().item()),
            )

            # Update convergence and exit checks after step (record step index)
            if self._should_exit(fmax_per_conf, after_step=True):
                logger.info("Exiting after step {}: converged={}", self.nsteps, self._converged)
                self._finalize_trajectories()
                return self._converged

    def step(self, forces: torch.Tensor) -> None:
        """Perform one BFGS update for each unconverged conformer.

        Skips conformers already marked as converged.
        """
        for idx, (start, end) in self._iter_conformer_slices():
            if hasattr(self.batch, "converged") and bool(self.batch.converged[idx].item()):
                logger.debug("Skipping conformer {} (already converged).", idx)
                continue

            pos_i = self.batch.pos[start:end]  # [Ni, 3]
            f_i = forces[start:end]  # [Ni, 3]

            # Flatten to 3Ni for BFGS math and use float64 for stability
            r = pos_i.reshape(-1).to(torch.float64)
            f = f_i.reshape(-1).to(torch.float64)

            d = r.numel()
            H = self._H.get(idx)
            r0 = self._r0.get(idx)
            f0 = self._f0.get(idx)

            if H is None:
                H = torch.eye(d, dtype=torch.float64, device=self.device) * 70.0
                logger.debug("Init Hessian for conformer {} with shape {}.", idx, tuple(H.shape))
            else:
                # Update Hessian using a BFGS-like rule
                dr = r - r0
                if dr.abs().max() >= 1e-7:
                    df = f - f0
                    a = torch.dot(dr, df)
                    dg = H @ dr
                    b = torch.dot(dr, dg)

                    # Safeguard denominators
                    eps = 1e-12
                    if torch.abs(a) <= eps:
                        a = (
                            torch.sign(a) * eps
                            if a != 0
                            else torch.tensor(eps, dtype=torch.float64, device=self.device)
                        )
                    if torch.abs(b) <= eps:
                        b = (
                            torch.sign(b) * eps
                            if b != 0
                            else torch.tensor(eps, dtype=torch.float64, device=self.device)
                        )

                    H = H - torch.outer(df, df) / a - torch.outer(dg, dg) / b
                    logger.debug("Updated Hessian for conformer {}.", idx)
                else:
                    logger.debug(
                        "Small displacement for conformer {}; skipping Hessian update.", idx
                    )

            # Determine step: dr = V @ ((f @ V) / |omega|)
            omega, V = torch.linalg.eigh(H)
            denom = torch.clamp(torch.abs(omega), min=1e-12)
            fV = f @ V  # [d]
            dr_flat = V @ (fV / denom)  # [d]
            dr = dr_flat.view(-1, 3).to(self.dtype)

            # Apply max step scaling
            steplengths = torch.linalg.vector_norm(dr, dim=1)  # [Ni]
            maxsteplength = torch.max(steplengths)
            if maxsteplength >= self.max_step:
                scale = (self.max_step / (maxsteplength + 1e-12)).to(self.dtype)
                dr = dr * scale
                logger.debug(
                    "Scaled step for conformer {} by factor {:.6f} (max steplength {:.6f}).",
                    idx,
                    float(scale.item()),
                    float(maxsteplength.item()),
                )

            # Update positions and carry state
            pos_i.add_(dr)
            self._H[idx] = H
            self._r0[idx] = r.detach().clone()
            self._f0[idx] = f.detach().clone()
            logger.debug("Applied step to conformer {}.", idx)

    # --------------------- utilities ---------------------

    def _check_batch(self) -> None:
        """Validate and augment the Batch/Data object for optimisation.

        - Ensures .pos shape and dtype.
        - Adds .ptr and .batch if given a single Data object.
        - Initializes batch.converged and batch.converged_step arrays on device.
        """
        if not hasattr(self.batch, "pos"):
            raise ValueError("Batch/Data must have a .pos tensor of shape [n_atoms, 3].")
        if self.batch.pos.ndim != 2 or self.batch.pos.size(-1) != 3:
            raise ValueError(f"pos must have shape [n_atoms, 3], got {tuple(self.batch.pos.shape)}")
        if not isinstance(self.batch.pos, torch.Tensor):
            raise ValueError("pos must be a torch.Tensor.")
        if self.batch.pos.dtype not in (torch.float32, torch.float64):
            raise ValueError("pos must be float32 or float64.")

        # If Data (single conformer), synthesize a 1-conformer view for iteration
        if not hasattr(self.batch, "ptr"):
            self.batch.ptr = torch.tensor([0, self.n_atoms], device=self.device, dtype=torch.long)
            self.batch.batch = torch.zeros(self.n_atoms, device=self.device, dtype=torch.long)
            logger.debug("Synthesized ptr/batch for single Data with natoms={}.", self.n_atoms)

        # Init per-conformer convergence tracking
        nconf = int(self.batch.ptr.numel() - 1)
        if (
            not hasattr(self.batch, "converged")
            or self.batch.converged is None
            or self.batch.converged.numel() != nconf
        ):
            self.batch.converged = torch.zeros(nconf, dtype=torch.bool, device=self.device)
            logger.debug("Initialized batch.converged with shape [{}].", nconf)
        if (
            not hasattr(self.batch, "converged_step")
            or self.batch.converged_step is None
            or self.batch.converged_step.numel() != nconf
        ):
            self.batch.converged_step = torch.full(
                (nconf,), -1, dtype=torch.long, device=self.device
            )
            logger.debug("Initialized batch.converged_step with shape [{}].", nconf)

        logger.debug("Batch check complete: natoms={}, nconf={}.", self.n_atoms, nconf)

    def _iter_conformer_slices(self):
        """Yield (idx, (start, end)) atom index slices for each conformer.

        Note: slices update batch.pos inplace. Masks create copies, so do not work here."""
        for i in range(self.batch.n_conformers):
            yield i, (int(self.batch.ptr[i].item()), int(self.batch.ptr[i + 1].item()))

    def _forces(self) -> torch.Tensor:
        """Request forces from the attached calculator and validate shape/dtype."""
        # f = self.calculator.forces(self.batch)  # TODO: uncomment
        f = torch.rand_like(self.batch.pos) / (5 * self.nsteps + 1)  # Dummy forces for testing
        if not isinstance(f, torch.Tensor):
            raise TypeError("calculator.forces(batch) must return a torch.Tensor.")
        if f.shape != batch.pos.shape:
            raise ValueError(
                f"forces shape must match pos shape; "
                f"got {tuple(f.shape)} vs {tuple(batch.pos.shape)}"
            )
        out = f.to(self.device, dtype=self.dtype)
        logger.debug("Forces computed with shape {} and dtype {}.", tuple(out.shape), out.dtype)
        return out

    def _per_conformer_max_force(self, forces: torch.Tensor) -> torch.Tensor:
        """Compute per-conformer max |F|.

        Returns:
            Tensor of shape [n_conformers] with each conformer's max per-atom force norm.
        """
        norms = torch.linalg.vector_norm(forces, dim=1)
        vals = []
        for i in range(batch.n_conformers):
            vals.append(norms[self.batch.batch == i].max())
        out = torch.stack(vals)
        return out

    def _update_convergence(self, fmax_per_conf: torch.Tensor, after_step: bool) -> None:
        """Update per-conformer convergence mask and converged_step.

        Args:
            fmax_per_conf: Tensor [n_conformers] with max |F| per conformer.
            after_step: If True, record converged_step as current step for newly converged.
        """
        if self.fmax is None:
            return
        target = float(self.fmax)
        newly_converged = (~self.batch.converged) & (fmax_per_conf < target)
        if newly_converged.any():
            idxs = torch.nonzero(newly_converged, as_tuple=False).view(-1)
            logger.info(
                "Newly converged conformers ({}): {} at step {}.",
                idxs.numel(),
                idxs.tolist(),
                self.nsteps if after_step else "pre-step",
            )
            # Update mask
            self.batch.converged = self.batch.converged.clone()
            self.batch.converged[newly_converged] = True
            # Record step index only after a step has been taken
            if after_step:
                self.batch.converged_step = self.batch.converged_step.clone()
                self.batch.converged_step[idxs] = self.nsteps

    def _should_exit(self, fmax_per_conf: torch.Tensor, after_step: bool) -> bool:
        """Update convergence bookkeeping and decide whether to stop.

        Exit if:
        - All conformers have converged,
        - Any conformer exceeds fexit,
        - Step limit reached.
        """
        # Update per-conformer convergence bookkeeping
        self._update_convergence(fmax_per_conf, after_step=after_step)

        # All converged
        if self.batch.converged.all().item():
            self._converged = True
            logger.info("All conformers converged at step {}.", self.nsteps)
            return True

        # Explosion (all non-converged conformers exceed fexit)
        if self.fexit is not None:
            active = ~self.batch.converged
            if active.any() and torch.all(fmax_per_conf[active] > float(self.fexit)):
                offenders = torch.nonzero(active, as_tuple=False).view(-1).tolist()
                logger.warning(
                    "Exiting due to fexit. All non-converged conformers exceeded fexit. "
                    "Offenders: {}.",
                    offenders,
                )
                self._converged = False
                return True

        # Step limit
        if self.steps >= 0 and self.nsteps >= self.steps:
            logger.info("Step limit reached: {} steps.", self.steps)
            self._converged = False
            return True

        return False

    def _finalize_trajectories(self) -> None:
        """Assemble batch.pos_dt [T, N, 3] and batch.pos_min [N, 3] on exit.

        - pos_dt contains coordinates after each step.
        - pos_min is constructed per conformer from its converged step if available,
          otherwise its final geometry.
        """
        # Ensure pos_dt exists and is on correct device/dtype
        if not hasattr(self.batch, "pos_dt") or self.batch.pos_dt is None:
            self.batch.pos_dt = torch.empty(
                (0, self.n_atoms, 3), dtype=self.dtype, device=self.device
            )
        else:
            self.batch.pos_dt = self.batch.pos_dt.to(device=self.device, dtype=self.dtype)

        frames = int(self.batch.pos_dt.shape[0])
        logger.debug("pos_dt assembled with {} frames.", frames)

        # pos_min: default to final geometry (last frame if available)
        if frames == 0:
            pos_min = self.batch.pos.detach().clone()
        else:
            pos_min = self.batch.pos_dt[-1].detach().clone()

        # Overwrite converged conformers with geometry at their converged step
        if hasattr(self.batch, "converged_step") and self.batch.converged_step is not None:
            replaced = []
            for i in range(self.batch.n_conformers):
                cs = int(self.batch.converged_step[i].item())
                if cs > 0 and frames >= cs:
                    if frames > 0:
                        # TODO: mask won't work here, mask create a copy, not change in-place
                        pos_min[batch.batch == i] = self.batch.pos_dt[self.nsteps, batch.batch == i]
                        replaced.append(int(i))
            if replaced:
                logger.debug("pos_min updated from converged steps for conformers: {}.", replaced)

        self.batch.pos_min = pos_min
        nconf = int(self.batch.ptr.numel() - 1)
        nconv = int(self.batch.converged.sum().item()) if hasattr(self.batch, "converged") else 0
        logger.info("Finalized trajectories: nconf={}, converged={}/{}.", nconf, nconv, nconf)


if __name__ == "__main__":
    from ase.build import molecule

    from ase_bfgs_batched.conformers import ConformerBatch

    atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
    batch = ConformerBatch.from_ase(atoms_list)

    optimiser = BFGSBatched(steps=10, fmax=0.05, fexit=500.0)
    optimiser.calculator = 0  # Replace with actual calculator implementing forces(batch)
    converged = optimiser.run(batch)
