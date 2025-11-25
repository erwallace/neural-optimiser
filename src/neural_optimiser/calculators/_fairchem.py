from typing import Literal

import torch
from torch_geometric.data import Batch, Data

from neural_optimiser.calculators.base import Calculator

try:
    if hasattr(torch, "serialization"):
        torch.serialization.add_safe_globals([slice])
except Exception:
    pass


class FAIRChemCalculator(Calculator):
    def __init__(
        self,
        model_paths: str,
        device: str = "cpu",
        task_name: str = "omol",
        default_dtype: Literal["float32", "float64"] = "float32",
    ):
        try:
            import fairchem  # noqa: F401
            from fairchem.core.units.mlip_unit import load_predict_unit
        except ImportError:
            raise ImportError(
                "FAIRChem is not installed. Run `uv pip install fairchem-core` to install."
            )
        self.device = device
        self.task_name = task_name
        self.default_dtype = torch.float32 if default_dtype == "float32" else torch.float64
        self.model_paths = model_paths
        self.predictor = load_predict_unit(path=model_paths, device=device)

    def __repr__(self):
        return (
            f"FAIRChemCalculator(model_paths={self.model_paths}, device={self.device}, "
            f"default_dtype={self.default_dtype}, task_name={self.task_name})"
        )

    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energies and forces for a batch of conformers using a FAIRChem model."""
        from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

        atoms = batch.to_ase()
        [
            atoms[i].info.update({"charge": batch.charge[i].item(), "spin": batch.spin[i].item()})
            for i in range(batch.n_conformers)
        ]
        atomic_data = [
            AtomicData.from_ase(
                atom, task_name="omol", r_data_keys=["charge", "spin"], r_edges=False
            )
            for atom in atoms
        ]

        batch = atomicdata_list_to_batch(atomic_data)
        output = self.predictor.predict(batch)
        return output["energy"], output["forces"]

    def get_energies(self, batch: Data | Batch) -> torch.Tensor:
        """Compute energies for a batch of conformers using a FAIRChem model."""
        energies, _ = self._calculate(batch)
        return energies
