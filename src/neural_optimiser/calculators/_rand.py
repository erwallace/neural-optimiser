import torch
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator


class RandomCalculator(Calculator):
    def __init__(self, seed: int = 42):
        self.device = None
        torch.manual_seed(seed)

    def _calculate(self, batch: Data | Batch) -> torch.Tensor:
        """A dummy calculator that returns random forces."""
        energies = torch.zeros(batch.n_conformers, device=self.device)
        forces = torch.randn_like(batch.pos, device=self.device)

        return energies, forces

    def to_atomic_data():
        """AtomicData not used."""
        pass
