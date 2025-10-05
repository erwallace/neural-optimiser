import torch
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator


class FAIRChemCalculator(Calculator):
    def __init__(self, model_path):
        try:
            from fairchem import FAIRChemCalculator as FAIRChemCalc
        except ImportError:
            raise ImportError(
                "MACE is not installed. Run `uv pip install fairchem-core` to install."
            )
        self.calculator = FAIRChemCalc(model_path)

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("FAIRChemCalculator is not yet implemented.")

    def to_atomic_data():
        pass
