import torch
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator


class FAIRChemCalculator(Calculator):
    def __init__(self, model_path):
        try:
            from fairchem import FAIRChemCalculator as FAIRChemCalc
        except ImportError:
            raise ImportError("MACE is not installed. Please install it to use MACECalculator.")
        self.calculator = FAIRChemCalc(model_path)

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        atoms = self.to_atomic_data(batch)
        return self.calculator.calculate(atoms)

    def to_atomic_data():
        pass

    def from_atomic_data():
        pass
