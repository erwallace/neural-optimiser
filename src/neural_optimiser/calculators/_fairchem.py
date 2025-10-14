import torch
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator


class FAIRChemCalculator(Calculator):
    def __init__(
        self, model_paths: str, device: str = "cpu", radius: float = 6.0, max_neighbours: int = 32
    ):
        try:
            import fairchem  # noqa: F401
            from fairchem.core.units.mlip_unit import load_predict_unit
        except ImportError:
            raise ImportError(
                "MACE is not installed. Run `uv pip install fairchem-core` to install."
            )
        self.device = device
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.model_paths = model_paths
        self.predictor = load_predict_unit(path=model_paths, device=device)

    def __repr__(self):
        return (
            f"FAIRChemCalculator(model_paths={self.model_paths}, device={self.device}, "
            f"max_neighbours={self.max_neighbours}, radius={self.radius})"
        )

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces for a batch of conformers using a FAIRChem model."""
        raise NotImplementedError("FAIRChemCalculator is not yet implemented.")

    def get_energies(self, batch: Data | Batch) -> torch.Tensor:
        raise NotImplementedError("FAIRChemCalculator is not yet implemented.")

    def to_atomic_data(self, batch: Data | Batch) -> Batch:
        raise NotImplementedError("FAIRChemCalculator is not yet implemented.")
