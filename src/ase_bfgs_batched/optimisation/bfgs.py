from typing import Any

from ase.optimize import BFGS
from torch_geometric.data import Batch, Data


class BFGSBatched(BFGS):
    """BFGS optimiser with exit conditions for strain relief.

    Exit conditions:
    1. Maximum force on any atom > fexit (dynamics exploding).
    2. Number of steps exceeds max_steps.
    3. Forces have converged (max force < fmax).
    """

    def __init__(
        self,
        max_step: float = 0.04,
        steps: int = -1,
        fmax: float | None = None,
        fexit: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.fexit = fexit
        self.steps = steps
        self.max_step = max_step
        self.calculator = None
        super().__init__(fmax=fmax, **kwargs)

    def __post_init__(self):
        if self.steps == -1 and not self.fmax:
            raise ValueError("Either fmax or steps must be set to define convergence.")

    def run(self, batch: Data | Batch, steps: int = -1) -> None:
        # if self.max_steps = -1, run until convergence

        if self.calculator is None:
            raise AttributeError("BFGSBatched.calculator must be set before running dynamics.")

        self._check_batch(batch)
        self._run()

    def _run(self) -> None:
        pass

    def step(self, batch: Data | Batch) -> None:
        pass

    def _check_batch(batch):
        # check that the batch has correct attributues
        pass


if __name__ == "__main__":
    dyn = BFGSBatched()
    dyn.run()
