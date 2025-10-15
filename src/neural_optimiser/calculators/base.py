from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Batch, Data


class Calculator(ABC):
    """Abstract base class for calculators used in neural optimisers."""

    device: str | None = None

    def __call__(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate inputs, set device, and delegate to implementation."""
        if not hasattr(batch, "pos") or not hasattr(batch, "atom_types"):
            raise ValueError("Batch must have 'pos' and 'atom_types' attributes.")
        self.device = batch.pos.device
        return self._calculate(batch)

    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def get_energies(self, batch: Data | Batch) -> torch.Tensor:
        """Get only energies from the calculator."""
        ...

    @abstractmethod
    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (energies, forces) from the calculator."""
        ...

    @abstractmethod
    def to_atomic_data(self, batch: Data | Batch) -> Batch:
        """Convert to AtomicData format compatible with ML model used."""
        ...
