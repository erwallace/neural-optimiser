import torch
from fairchem.core.units.mlip_unit import load_predict_unit
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator


class FAIRChemCalculator(Calculator):
    def __init__(
        self, model_paths: str, device: str = "cpu", radius: float = 6.0, max_neighbours: int = 32
    ):
        try:
            import fairchem  # noqa: F401
            from fairchem.core.units.mlip_unit import load_predict_unit  # noqa: F401
        except ImportError:
            raise ImportError(
                "MACE is not installed. Run `uv pip install fairchem-core` to install."
            )
        self.device = device
        self.radius = radius
        self.max_neighbours = max_neighbours
        self.model_paths = model_paths
        # self.predictor = load_predict_unit(path=model_paths, device=device)

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


if __name__ == "__main__":
    from ase.build import molecule
    from fairchem.core import FAIRChemCalculator as fair_calc
    from fairchem.core import pretrained_mlip

    try:
        import torch

        if hasattr(torch, "serialization"):
            torch.serialization.add_safe_globals([slice])
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_paths = "./models/omol25_esen_sm_direct.pt"

    # EXAMPLE 1
    predictor = load_predict_unit(path=model_paths, device=device)
    calculator = fair_calc(predictor)

    # Create ASE molecule and assign FAIRChem calculator
    atoms = molecule("H2O")
    atoms.info = {"charge": 0, "spin": 1}
    atoms.calc = calculator
    print(atoms.get_potential_energy())

    # EXAMPLE 2
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
    # get_predict_unit looks up ckpt and calls load_predict_unit.
    calculator2 = fair_calc(predictor, task_name="omol")

    # Create ASE molecule and assign FAIRChem calculator
    atoms = molecule("H2O")
    atoms.info = {"charge": 0, "spin": 1}
    atoms.calc = calculator2
    print(atoms.get_potential_energy())
