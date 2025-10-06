from abc import ABC, abstractmethod

import torch
from loguru import logger
from torch_geometric.data import Batch, Data


class Optimiser(ABC):
    """Optimiser base class with additional exit conditions, operating on a PyG Batch/Data.

    - Maintains one Hessian per conformer in the batch.
    - Exits on convergence (fmax), explosion (fexit) or step limit.
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
        """Create a batched optimiser.

        Args:
            max_step: Maximum allowed displacement per-atom per step.
            steps: Maximum number of steps. If -1, only fmax/fexit control exit.
            fmax: Convergence threshold on max per-atom force norm.
            fexit: Early-exit threshold if all non-converged max per-conformer forces exceeds this
            value.
        """
        self.fexit = fexit
        self.steps = steps
        self.max_step = max_step
        self.fmax = fmax
        self.calculator = None

        self.nsteps: int = 0
        self._converged: bool = False
        self._n_fmax: float = 0.0

        logger.debug(
            f"Initialized {self.__class__.__name__}(max_step={max_step}, steps={steps}, "
            f"fmax={fmax}, fexit={fexit})"
        )

        if self.steps == -1 and self.fmax is None:
            raise ValueError("Either fmax or steps must be set to define convergence.")

    def run(self, batch: Data | Batch) -> bool:
        """Run until exit conditions are met.

        Returns:
            True if all conformers converged, else False.
        """
        if self.calculator is None:
            raise AttributeError(
                f"{self.__class__.__name__}.calculator must be set before running dynamics."
            )

        self.device = batch.pos.device
        self.dtype = batch.pos.dtype
        self.n_atoms = batch.n_atoms
        self.n_confs = batch.n_conformers
        self.batch = batch

        self._check_batch()

        logger.info(
            f"Starting {self.__class__.__name__}: nconf={self.n_confs}, natoms={self.n_atoms}, "
            f"steps={self.steps}, fmax={self.fmax}, fexit={self.fexit}, max_step={self.max_step}"
        )
        return self._run()

    # --------------------- internal API ---------------------

    def _run(self) -> bool:
        """Internal driver loop. Computes forces, steps, and exit handling."""
        self.nsteps = 0
        self._converged = False

        # Init per-step trajectory storage directly on batch: [T, N, 3]
        self.batch.pos_dt = self.batch.pos.clone().unsqueeze(dim=0)
        self.batch.forces_dt = torch.empty(
            (0, self.n_atoms, 3), device=self.device, dtype=self.dtype
        )
        self.batch.energies_dt = torch.empty(
            (0, self.n_confs), device=self.device, dtype=self.dtype
        )

        # Initial force evaluation
        energies, forces = self._forces()
        fmax_per_conf = self._per_conformer_max_force(forces)

        # Update per-conformer converged mask (no step index recorded yet)
        self._update_convergence(fmax_per_conf, after_step=False)

        # Exit checks before any step
        if self._should_exit(fmax_per_conf, after_step=False):
            logger.info(
                f"Exiting before any step: converged={self._converged}, nsteps={self.nsteps}"
            )
            self._finalise_trajectories()
            return self._converged

        while True:
            with torch.no_grad():
                self.step(forces)

            # Append current positions as a new frame to batch.pos_dt
            self.batch.pos_dt = torch.cat(
                (self.batch.pos_dt, self.batch.pos.unsqueeze(0)),
                dim=0,
            )
            self.nsteps += 1

            energies, forces = self._forces()
            fmax_per_conf = self._per_conformer_max_force(forces)

            # Update convergence and exit checks after step (record step index)
            if self._should_exit(fmax_per_conf, after_step=True):
                logger.info(f"Exiting after step {self.nsteps}: converged={self._converged}")
                self._finalise_trajectories()
                return self._converged

    @abstractmethod
    def step(self) -> None:
        """Perform one optimisation step."""
        ...

    # --------------------- utilities ---------------------

    def _check_batch(self) -> None:
        """Validate and augment the Batch/Data object for optimisation.

        - Ensures .pos shape and dtype.
        - Adds .ptr and .batch if given a single Data object.
        - Initializes batch.converged and batch.converged_step arrays on device.
        """
        if not hasattr(self.batch, "pos"):
            raise ValueError("Batch/Data must have a .pos tensor of shape [n_atoms, 3].")
        if not isinstance(self.batch.pos, torch.Tensor):
            raise ValueError("pos must be a torch.Tensor.")
        if self.batch.pos.ndim != 2 or self.batch.pos.size(-1) != 3:
            raise ValueError(f"pos must have shape [n_atoms, 3], got {tuple(self.batch.pos.shape)}")
        if self.batch.pos.dtype not in (torch.float32, torch.float64):
            raise ValueError("pos must be float32 or float64.")

        # If Data (single conformer), synthesize a 1-conformer view for iteration
        if not hasattr(self.batch, "ptr"):
            self.batch.ptr = torch.tensor([0, self.n_atoms], device=self.device, dtype=torch.long)
            self.batch.batch = torch.zeros(self.n_atoms, device=self.device, dtype=torch.long)

        # Init per-conformer convergence tracking
        if (
            not hasattr(self.batch, "converged")
            or self.batch.converged is None
            or self.batch.converged.numel() != self.n_confs
        ):
            self.batch.converged = torch.zeros(self.n_confs, dtype=torch.bool, device=self.device)
        if (
            not hasattr(self.batch, "converged_step")
            or self.batch.converged_step is None
            or self.batch.converged_step.numel() != self.n_confs
        ):
            self.batch.converged_step = torch.full(
                (self.n_confs,), -1, dtype=torch.long, device=self.device
            )

    def _iter_conformer_slices(self):
        """Yield (idx, (start, end)) atom index slices for each conformer.

        Note: slices update batch.pos inplace. Masks create copies, so do not work here."""
        for i in range(self.n_confs):
            yield i, (int(self.batch.ptr[i].item()), int(self.batch.ptr[i + 1].item()))

    def _forces(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Request energies and forces from the attached calculator."""
        e, f = self.calculator.calculate(self.batch)

        self.batch.energies_dt = torch.cat((self.batch.energies_dt, e.unsqueeze(0)), dim=0)
        self.batch.forces_dt = torch.cat((self.batch.forces_dt, f.unsqueeze(0)), dim=0)

        return e, f

    def _per_conformer_max_force(self, forces: torch.Tensor) -> torch.Tensor:
        """Compute per-conformer max |F|.

        Returns:
            Tensor of shape [n_conformers] with each conformer's max per-atom force norm.
        """
        norms = torch.linalg.vector_norm(forces, dim=1)
        vals = []
        for i in range(self.n_confs):
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
            # Update mask
            self.batch.converged[newly_converged] = True
            # Record step index only after a step has been taken
            if after_step:
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

        if self.nsteps % 10 == 0:
            logger.info(
                f"Step {self.nsteps}: {int(self.batch.converged.sum().item())}/{self.n_confs} "
                "conformers converged."
            )

        # All converged
        if self.batch.converged.all().item():
            self._converged = True
            logger.info(f"All conformers converged by step {self.nsteps}.")
            return True

        # Explosion (all non-converged conformers exceed fexit)
        if self.fexit is not None:
            active = ~self.batch.converged
            if active.any() and torch.all(fmax_per_conf[active] > float(self.fexit)):
                offenders = torch.nonzero(active, as_tuple=False).view(-1).tolist()
                logger.warning(
                    f"Exiting due to fexit. All non-converged conformers exceeded fexit. "
                    f"Offenders: {offenders}."
                )
                self._converged = False
                return True

        # Step limit
        if self.steps >= 0 and self.nsteps >= self.steps:
            logger.info(f"Step limit reached: {self.steps} steps.")
            self._converged = False
            return True

        return False

    def _finalise_trajectories(self) -> None:
        """Assemble batch.pos_dt [T, N, 3] and batch.pos_min [N, 3] on exit.

        - pos_dt contains coordinates after each step.
        - pos is constructed per conformer from its converged step if available,
          otherwise its final geometry.
        """
        # Return energeis, forces, positions of the converged step if available, else final
        converged_steps_by_atom = self.batch.converged_step[self.batch.batch]
        atom_idx = torch.arange(converged_steps_by_atom.numel(), device=self.device)
        conformer_idx = torch.arange(self.batch.n_conformers, device=self.device)
        self.batch.pos = self.batch.pos_dt[converged_steps_by_atom, atom_idx]
        self.batch.forces = self.batch.forces_dt[converged_steps_by_atom, atom_idx]
        self.batch.energies = self.batch.energies_dt[self.batch.converged_step, conformer_idx]

        nconv = int(self.batch.converged.sum().item())
        logger.info(
            f"Finalized trajectories: steps={self.nsteps}, nconfs={self.n_confs}, "
            f"converged={nconv}/{self.n_confs}."
        )
