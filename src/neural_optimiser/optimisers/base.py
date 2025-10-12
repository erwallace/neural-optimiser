from abc import ABC, abstractmethod

import torch
from loguru import logger
from torch_geometric.data import Batch, Data


class Optimiser(ABC):
    """Abstract base class for batched molecular geometry optimisation.

    This class provides the core framework for running optimisation simulations on a
    `torch_geometric.data.Batch` or `Data` object. It handles the main optimisation
    loop, convergence checks, and trajectory logging. Subclasses must implement the
    `step` method to define the specific optimisation algorithm.

    The optimiser will stop under one of the following conditions:
    - All conformers in the batch converge (maximum norm force per atom is below `fmax`).
    - The maximum number of steps (`steps`) is reached.
    - The forces on all non-converged conformers exceed the `fexit` threshold,
      indicating a potential explosion or instability.

    After the run, the input `batch` object is updated with the results, including
    the full trajectory and the final converged geometries.

    Attributes:
        max_step (float): The maximum distance any single atom can move in one step.
        steps (int): The maximum number of optimisation steps to perform. If -1,
            the optimisation runs until `fmax` or `fexit` criteria are met.
        fmax (float | None): The force convergence threshold. A conformer is
            considered converged if the maximum norm force on all of its atoms is
            less than this value.
        fexit (float | None): The explosion threshold. The optimisation will
            terminate early if the max force on all non-converged conformers
            exceed this value.
        calculator (Calculator | None): The calculator instance used to compute
            energies and forces for the molecular geometries. This must be set
            before calling `run`.
        nsteps (int): The number of steps taken in the most recent optimisation run.
        trajectory (Trajectory): The trajectory object that stores positional,
            force, and energy data for each step.
    """

    def __init__(
        self,
        max_step: float = 0.04,
        steps: int = -1,
        fmax: float | None = None,
        fexit: float | None = None,
    ) -> None:
        """Initialises a new batched optimiser.

        Args:
            max_step: Maximum allowed displacement for any atom in a single step.
                Used to control the step size and prevent overly large changes in
                geometry.
            steps: The maximum number of optimisation steps to perform. If set to
                -1, the optimisation will continue until either the `fmax` or
                `fexit` criteria are met.
            fmax: The convergence threshold for the maximum force on a single
                atom. If the maximum force on every atom in a conformer is below
                this value, the conformer is considered converged.
            fexit: An early-exit threshold. If the maximum force on
                all non-converged conformers exceeds this value, the optimisation
                is terminated to prevent instability.
        """
        self.fexit = fexit
        self.steps = steps
        self.max_step = max_step
        self.fmax = fmax
        self.calculator = None

        self.nsteps: int = 0
        self._converged: bool = False

        logger.debug(
            f"Initialized {self.__class__.__name__}(max_step={max_step}, steps={steps}, "
            f"fmax={fmax}, fexit={fexit})"
        )

        if self.steps == -1 and self.fmax is None:
            raise ValueError("Either fmax or steps must be set to define convergence.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(max_step={self.max_step}, steps={self.steps}, "
            f"fmax={self.fmax}, fexit={self.fexit})"
        )

    def run(self, batch: Data | Batch) -> bool:
        """Runs the optimisation until a termination condition is met.

        This method executes the main optimisation loop. It iteratively calls the
        `step` method (implemented by subclasses) and updates the state of the
        `batch` object with trajectory and convergence information.

        The `batch` object is modified in-place to contain the results.

        Args:
            batch: A `torch_geometric.data.Batch` or `Data` object containing
                the initial molecular geometries. It must have `pos` and `atom_types`
                attributes.

        Returns:
            True if all conformers in the batch converged successfully, False otherwise.

        Raises:
            AttributeError: If `self.calculator` has not been set before calling `run`.
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
        """Internal driver loop that computes forces, steps, and handles exit conditions."""
        self.nsteps = 0
        self._converged = False
        self.trajectory = Trajectory(self.batch)

        # Initial force evaluation
        energies, forces = self.calculator(self.batch)
        self.trajectory.add_initial_properties(energies, forces)
        fmax_per_conf = self._per_conformer_max_force(forces)

        # Update per-conformer converged mask (no step index recorded yet)
        self._update_convergence(fmax_per_conf, after_step=False)

        # Exit checks before any step
        if self._should_exit(fmax_per_conf, after_step=False):
            logger.info(
                f"Exiting before any step: converged={self._converged}, nsteps={self.nsteps}"
            )
            self.trajectory.finalise(self.batch)
            return self._converged

        while True:
            with torch.no_grad():
                self.step(forces)

            self.nsteps += 1

            # Get new forces and energies and update trajectory
            energies, forces = self.calculator(self.batch)
            self.trajectory.add_frame(self.batch.pos, energies, forces)
            fmax_per_conf = self._per_conformer_max_force(forces)

            # Update convergence and exit checks after step (record step index)
            if self._should_exit(fmax_per_conf, after_step=True):
                logger.info(f"Exiting after step {self.nsteps}: converged={self._converged}")
                self.trajectory.finalise(self.batch)
                return self._converged

    @abstractmethod
    def step(self, forces: torch.Tensor) -> None:
        """Performs a single optimisation step.

        This method must be implemented by subclasses to define the logic for
        updating atom positions based on the current forces.

        Args:
            forces: A tensor of shape `(n_atoms, 3)` containing the forces on each atom.
        """
        ...

    # --------------------- utilities ---------------------

    def _check_batch(self) -> None:
        """Validates and prepares the Batch/Data object for optimisation.

        This method performs the following actions:
        - Ensures the `.pos` attribute exists, is a tensor, and has the correct shape and dtype.
        - If a single `Data` object is provided, it synthesizes `.ptr` and `.batch`
          attributes to treat it as a batch with one conformer.
        - Initializes the `batch.converged` and `batch.converged_step` tensors on the
          correct device for tracking convergence status.
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
        """Yields atom index slices for each conformer in the batch.

        This is a generator that provides the start and end indices for the atoms
        belonging to each conformer, which is useful for per-conformer operations.

        Yields:
            A tuple of (conformer_index, (start_atom_index, end_atom_index)).
        """
        for i in range(self.n_confs):
            yield i, (int(self.batch.ptr[i].item()), int(self.batch.ptr[i + 1].item()))

    def _per_conformer_max_force(self, forces: torch.Tensor) -> torch.Tensor:
        """Computes the maximum per-atom force norm for each conformer.

        Args:
            forces: A tensor of shape `(n_atoms, 3)` containing the forces for the entire batch.

        Returns:
            A tensor of shape `(n_conformers,)` where each element is the maximum
            force norm for the corresponding conformer.
        """
        norms = torch.linalg.vector_norm(forces, dim=1)
        vals = []
        for i in range(self.n_confs):
            vals.append(norms[self.batch.batch == i].max())
        out = torch.stack(vals)
        return out

    def _update_convergence(self, fmax_per_conf: torch.Tensor, after_step: bool) -> None:
        """Updates the convergence status for each conformer.

        This method checks which conformers have a maximum force below `self.fmax`
        and updates their status in `batch.converged` and `batch.converged_step`.

        Args:
            fmax_per_conf: A tensor of shape `(n_conformers,)` with the maximum
                force for each conformer.
            after_step: A boolean indicating whether this check is performed after an
                optimisation step. If True, the current step index is recorded for
                newly converged conformers.
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
        """Determines if the optimisation should terminate.

        Checks for the three exit conditions:
        1. All conformers have converged.
        2. The step limit has been reached.
        3. All non-converged conformers have forces exceeding `fexit`.

        Args:
            fmax_per_conf: A tensor of shape `(n_conformers,)` with the maximum
                force for each conformer.
            after_step: A boolean passed to `_update_convergence`.

        Returns:
            True if the optimisation should exit, False otherwise.
        """
        # Update per-conformer convergence bookkeeping
        self._update_convergence(fmax_per_conf, after_step=after_step)

        if self.nsteps % 10 == 0 and self.nsteps > 0:
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


class Trajectory:
    """Handles the storage and finalisation of optimisation trajectory data."""

    def __init__(self, batch: Data | Batch):
        """Initialise trajectory with the starting geometry."""
        self.device = batch.pos.device
        self.n_confs = batch.n_conformers

        # Use lists for efficient appending of tensors
        self.pos_dt = [batch.pos.clone()]
        self.forces_dt = []
        self.energies_dt = []

    def add_frame(self, pos: torch.Tensor, energies: torch.Tensor, forces: torch.Tensor):
        """Append a new frame (positions, energies, forces) to the trajectory."""
        self.pos_dt.append(pos.clone())
        self.energies_dt.append(energies)
        self.forces_dt.append(forces)

    def add_initial_properties(self, energies: torch.Tensor, forces: torch.Tensor):
        """Store the energies and forces of the initial state."""
        self.energies_dt.append(energies)
        self.forces_dt.append(forces)

    def finalise(self, batch: Data | Batch):
        """
        Stack trajectory lists into tensors and attach them to the final batch object.

        - pos_dt, forces_dt, energies_dt contain the full trajectory.
        - pos, forces, energies contain the properties of the converged step
          if available, otherwise the final step.
        """
        # Attach full trajectories to the batch
        batch.pos_dt = torch.stack(self.pos_dt, dim=0)
        batch.forces_dt = torch.stack(self.forces_dt, dim=0)
        batch.energies_dt = torch.stack(self.energies_dt, dim=0)

        # Get properties from the converged step, or the final step for non-converged
        converged_steps_by_atom = batch.converged_step[batch.batch]
        atom_idx = torch.arange(converged_steps_by_atom.numel(), device=self.device)
        conformer_idx = torch.arange(self.n_confs, device=self.device)

        # For non-converged items, converged_step is -1, which correctly indexes the last frame.
        batch.pos = batch.pos_dt[converged_steps_by_atom, atom_idx]
        batch.forces = batch.forces_dt[converged_steps_by_atom, atom_idx]
        batch.energies = batch.energies_dt[batch.converged_step, conformer_idx]
