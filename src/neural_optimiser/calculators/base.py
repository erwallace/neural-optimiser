from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Batch, Data


class Calculator(ABC):
    """Abstract base class for calculators used in neural optimisers."""

    device: str | None = None

    def calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate inputs, set device, and delegate to implementation."""
        if not hasattr(batch, "pos") or not hasattr(batch, "atom_types"):
            raise ValueError("Batch must have 'pos' and 'atom_types' attributes.")
        self.device = batch.pos.device
        return self._calculate(batch)

    @abstractmethod
    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (energies, forces)."""
        raise NotImplementedError

    @abstractmethod
    def to_atomic_data():
        """Convert to AtomicData format compatible with ML model used."""
        ...
