# neural-optimiser
Batched optimisation algorithms for neural network potential–driven molecular structure relaxation on top of PyTorch Geometric.

### Key features

- Batched per-conformer BFGS with per-atom max-step control.
- Early exit on convergence (`fmax`), explosion (`fexit`), or step cap (`steps`).
- Trajectory collection per step (`batch.pos_dt`) and converged geometries (`batch.pos_min`).
- IO methods for RDkit molecules and ASE atoms objects.

## Installation (uv)
Prerequisites: Python 3.11+, PyTorch and torch-geometric compatible with your environment.

Create a virtual environment and install the package:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Optional dev tools:
```bash
uv pip install -e ".[dev]"
uv run pre-commit install
```

**Note**: RDKit and torch-geometric may require platform-specific wheels. If uv/pip cannot resolve them directly, install those dependencies first using appropriate channels and then install this package.

## Quick Start

### Run a Batched BFGS Optimisation

This example uses `neural_optimiser.optimise._bfgs.BFGS`, and a dummy calculator, `neural_optimiser.calculators._random.RandomCalculator`.

```python
import torch
from ase.build import molecule

from neural_optimiser.optimise import BFGS
from neural_optimiser.calculators import RandomCalculator
from neural_optimiser.conformers import ConformerBatch

# Create a batch of molecules (each becomes a conformer)
atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
batch = ConformerBatch.from_ase(atoms_list, device="cpu")

# Configure optimiser and attach a calculator that provides forces
optimiser = BFGS(steps=200, fmax=0.05, fexit=500.0, max_step=0.04)
optimiser.calculator = RandomCalculator()

# Run optimisation
converged = optimiser.run(batch)
print("Converged:", converged)
print("Steps:", optimiser.nsteps)

# Trajectory [T, N, 3] and converged coordinates [N, 3]
print("pos_dt shape:", tuple(batch.pos_dt.shape))
print("pos_min shape:", tuple(batch.pos_min.shape))
```

**Notes:**
- `fmax` triggers convergence per conformer using the maximum per-atom force norm,. Ether `fmax` or `steps` must be specified.
- `fexit` triggers early exit if all non-converged conformers exceed the threshold.
- Trajectories are accumulated in memory as `batch.pos_dt`; converged geometries are indexed into `batch.pos_min` (final positions are returned for non-converged conformers). See `neural_optimiser.optimise.base.Optimiser` for more details.

### Using Your Own Calculator

Implement a calculator by subclassing `neural_optimiser.calculators.base.Calculator` and returning energies and forces for the full batch.

```python
import torch
from torch_geometric.data import Batch, Data
from neural_optimiser.calculators.base import Calculator

class MyCalculator(Calculator):
    def _calculate(self, batch: Data | Batch) -> tuple[torch.Tensor, torch.Tensor]:
        # energies: required shape [N_atoms]
        energies = torch.zeros(batch.n_conformers, device=self.device, dtype=torch.float32)
        # forces: required shape [N_atoms, 3] matching batch.pos
        forces = torch.zeros_like(batch.pos, device=self.device)
        # ... fill forces from your model ...
        return energies, forces

    def to_atomic_data():
        pass

    def from_atomic_data():
        pass
```

### Data Containers

**neural_optimiser.conformers._conformer.Conformer**

- Fields
  - `atom_types`: Long tensor [n_atoms]
  - `pos`: Float tensor [n_atoms, 3]
  - `smiles`: optional string
- Constructors
  - `from_ase(Atoms)`
  - `from_rdkit(Mol[, Conformer])`
- Converters
  - `to_ase()` -> `Atoms`
  - `to_rdkit()` -> `Mol` with one 3D conformer

**neural_optimiser.conformers._conformer_batch.ConformerBatch**

Extends torch-geometric Batch and groups many conformers.
- Constructors
  - `from_ase(list[Atoms])`
  - `from_rdkit(list[Mol] | Mol)`
  - `from_data_list(list[Conformer])`
- Properties
  - `n_molecules`
  - `n_conformers`
  - `n_atoms`
- Slicing
  - `conformer(i)` -> Conformer view referencing the i-th conformer’s atoms
- Additional indices
  - `batch`: per-atom conformer index
  - `molecule_idxs`: per-atom parent molecule index

## Testing
```bash
uv run pytest tests/
```

## License
Apache 2.0 - see [LICENSE](LICENSE).
