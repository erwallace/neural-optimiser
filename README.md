# neural-optimiser
Batched optimisation algorithms for neural network potential–driven molecular structure relaxation on top of PyTorch Geometric.

### Key features

- Batched per-conformer BFGS with per-atom max-step control.
- Early exit on convergence (`fmax`), explosion (`fexit`), or step cap (`steps`).
- Trajectory collection per step (`batch.pos_dt`, `batch.forces_dt`, `batch.energies_dt`) and converged properties (`batch.pos`, `batch.energies`, `batch.forces`).
- IO methods for RDkit molecules and ASE atoms objects.

## Installation

### Install from PyPi

```bash
pip install neural-optimiser

pip install neural-optimiser[mace]
```

### Install from source (uv)
Prerequisites: Python 3.12, PyTorch and torch-geometric compatible with your environment.

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

Installation for specific calculators:
```bash
uv run install-mace
```

**Note**: RDKit and torch-geometric may require platform-specific wheels. If uv/pip cannot resolve them directly, install those dependencies first using appropriate channels and then install this package.

## Quick Start

All of the code below and more is available in the [tutorial](notebooks/tutorial.ipynb).

### Run a Simple Batched BFGS Optimisation

This example uses `neural_optimiser.optimise._bfgs.BFGS`, and a dummy calculator, `neural_optimiser.calculators._rand.RandomCalculator`.

```python
from ase.build import molecule

from neural_optimiser.optimisers import BFGS
from neural_optimiser.calculators import RandomCalculator
from neural_optimiser.conformers import ConformerBatch

# Create a batch of molecules (each becomes a conformer)
atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
batch = ConformerBatch.from_ase(atoms_list, device="cpu")

# Configure optimiser and attach a calculator that provides forces
optimiser = BFGS(steps=10, fmax=0.05, fexit=500.0, max_step=0.04)
optimiser.calculator = RandomCalculator()

# Run optimisation
converged = optimiser.run(batch)
print("All Converged:", converged)
for i, (conv, nsteps) in enumerate(zip(batch.converged, batch.converged_step)):
    print(f"Conformer {i}: Converged: {conv}, On step {nsteps}")

# Trajectory [T, N, 3] and converged coordinates [N, 3]
print("pos_dt shape:", tuple(batch.pos_dt.shape))
print("pos shape:", tuple(batch.pos.shape))
```

**Notes:**
- `fmax` triggers convergence per conformer using the maximum per-atom force norm,. Ether `fmax` or `steps` must be specified.
- `fexit` triggers early exit if all non-converged conformers exceed the threshold.
- Trajectories are accumulated in memory as `batch.pos_dt` along with their energies and forces (`batch.forces_dt`, `batch.energies_dt`); converged geometries are indexed into `batch.pos`, `batch.energies`, `batch.forces` (final positions are returned for non-converged conformers). See `neural_optimiser.optimise.base.Optimiser` for more details.

### Run a Larger BFGS Optimisation using the ConformerDataLoader

For large datasets all your conformers may not fit in memory at once. In this case you can use the `neural_optimiser.datasets.base.ConformerDataLoader` to stream conformers in mini-batches.

```python
from rdkit import Chem
from rdkit.Chem import AllChem

from neural_optimiser.conformers import Conformer, ConformerBatch
from neural_optimiser.datasets.base import ConformerDataset, ConformerDataLoader
from neural_optimiser.optimisers import BFGS
from neural_optimiser.calculators import RandomCalculator

# Build a pool of conformers from RDKit molecules
smiles_list = ["CCO", "CC", "CCN"]
mols = []
for smiles in smiles_list:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(m, numConfs=10, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
    mols.append(m)

big_batch = ConformerBatch.from_rdkit(mols)  # creates one Conformer per RDKit conformer

# Dataset/DataLoader -> yields ConformerBatch
dataset = ConformerDataset(big_batch.to_data_list())
dataloader = ConformerDataLoader(dataset, batch_size=8, device="cpu", shuffle=True, num_workers=0)

# Configure optimiser and attach a calculator that provides forces
optimiser = BFGS(steps=10, fmax=0.05, fexit=500.0, max_step=0.04)
optimiser.calculator = RandomCalculator()

for batch in dataloader:
    optimiser.run(batch)
```

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

**Conformer**

Molecules with 3D geometries are stored as `neural_optimiser.conformers._conformer.Conformer` objects.

```python
from ase.build import molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from neural_optimiser.conformers import Conformer

# From ASE
atoms = molecule("H2O")
conf1 = Conformer.from_ase(atoms, smiles="O")

print(type(conf1).__name__)
print("atom_types:", conf1.atom_types.shape)  # [n_atoms]
print("pos:", conf1.pos.shape)                # [n_atoms, 3]
print("smiles:", conf1.smiles)

# Convert back to ASE
atoms2 = conf1.to_ase()
print("ASE atoms:", atoms2.get_chemical_formula(), atoms2.positions.shape)

# From RDKit
mol = Chem.MolFromSmiles("CCO")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())
conf2 = Conformer.from_rdkit(mol)

# Convert back to RDKit (returns a Mol with one 3D conformer)
mol2 = conf2.to_rdkit()
print("RDKit confs:", mol2.GetNumConformers())
```

**ConformerBatch**

Different molecules, or conformers of the same molecule (or both) can be stored in a `neural_optimiser.conformers._conformer_batch.ConformerBatch` object.

```python
from ase.build import molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from neural_optimiser.conformers import Conformer, ConformerBatch

# Build from ASE
atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
batch_ase = ConformerBatch.from_ase(atoms_list, device="cpu")
print("ASE batch:", batch_ase.n_molecules, batch_ase.n_conformers, batch_ase.n_atoms)

# Slice a single conformer view
conf0 = batch_ase.conformer(0)
print("conf0 pos:", conf0.pos.shape)

# Indices available on the batch (per-atom)
print("batch index shape:", batch_ase.batch.shape)
print("molecule_idxs shape:", batch_ase.molecule_idxs.shape)

# Build from RDKit (multiple conformers per molecule also supported)
def rdkit_with_coords(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, AllChem.ETKDG())
    return m

mol_list = [rdkit_with_coords("O"), rdkit_with_coords("CCO")]
batch_rd = ConformerBatch.from_rdkit(mol_list, device="cpu")
print("RDKit batch:", batch_rd.n_molecules, batch_rd.n_conformers, batch_rd.n_atoms)

# Build from a list of Conformer objects
c1 = Conformer.from_ase(molecule("H2O"))
c2 = Conformer.from_ase(molecule("NH3"))
batch_list = ConformerBatch.from_data_list([c1, c2], device="cpu")
print("Data list batch:", batch_list.n_molecules, batch_list.n_conformers, batch_list.n_atoms)
```

## Testing
```bash
uv run pytest tests/
```

## License
Apache 2.0 - see [LICENSE](LICENSE).
