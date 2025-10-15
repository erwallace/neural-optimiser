from typing import Any

import torch
from ase import Atoms
from loguru import logger
from rdkit import Chem
from torch_geometric.data import Batch

from neural_optimiser.conformers import Conformer


class ConformerBatch(Batch):
    """A batch of molecular conformers."""

    atom_type_dtype = torch.int64
    pos_dtype = torch.float32
    molecule_idx_dtype = torch.int64

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.__post_init__()
        self.to(device)

    def __post_init__(self):
        """Validate attributes."""

        if getattr(self, "atom_types", None) is None or getattr(self, "pos", None) is None:
            return  # placeholder instance during Batch.from_data_list
        if self.pos.ndim != 2 or self.pos.size(-1) != 3:
            raise ValueError(f"pos must have shape [n_atoms, 3], got {tuple(self.pos.shape)}")
        if self.atom_types.ndim != 1:
            raise ValueError(
                f"atom_types must be 1-D [n_atoms], got {tuple(self.atom_types.shape)}"
            )
        if self.atom_types.size(0) != self.pos.size(0):
            raise ValueError(
                f"atom_types and pos must have matching n_atoms, "
                f"got {self.atom_types.size(0)} vs {self.pos.size(0)}"
            )
        if self.atom_types.dtype != self.atom_type_dtype:
            raise ValueError(
                f"atom_types must have dtype {self.atom_type_dtype}, got {self.atom_types.dtype}"
            )
        if self.pos.dtype != self.pos_dtype:
            raise ValueError(f"pos must have dtype {self.pos_dtype}, got {self.pos.dtype}")

    @property
    def n_molecules(self) -> int:
        """Number of molecules in the batch."""
        return len(set(self.smiles)) if hasattr(self, "smiles") else None

    @property
    def n_conformers(self) -> int:
        """Number of conformers in the batch."""
        return self.batch.max().item() + 1

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the batch."""
        return self.pos.size(0)

    def conformer(self, idx: int, step: int | None = None) -> Conformer:
        """Get the i-th conformer in the batch at the n-th relaxation step.

        If step is not given and optimisation has been performed, the converged structure is
        returned.
        """
        kwargs: dict[str, Any] = {}

        for k, v in self.__dict__["_store"].items():
            if k in ["batch", "ptr", "device"]:
                continue
            elif torch.is_tensor(v) and v.dim() == 1 and v.size(0) == self.n_atoms:
                # e.g. atom_types
                kwargs[k] = v[self.batch == idx]
            elif torch.is_tensor(v) and v.dim() == 1 and v.size(0) == self.n_conformers:
                # e.g. energies, charges, spins
                kwargs[k] = v[idx]
            elif torch.is_tensor(v) and v.dim() == 2 and v.size(0) == self.n_atoms:
                # e.g. pos, forces
                kwargs[k] = v[self.batch == idx]
            elif torch.is_tensor(v) and v.dim() == 3 and v.size(1) == self.n_atoms:
                # e.g. pos_dt, forces_dt
                kwargs[k] = v[:, self.batch == idx, :]
            elif torch.is_tensor(v) and v.dim() == 2 and v.size(1) == self.n_conformers:
                # e.g. energies_dt
                kwargs[k] = v[:, idx]
            elif isinstance(v, list) and len(v) == self.n_conformers:
                # e.g. smiles
                kwargs[k] = v[idx]
            else:
                logger.warning(f"Attribute {k} not added to Conformer ({v}).")

        if step is not None:  # if optimisation has been performed
            if hasattr(self, "pos_dt"):
                kwargs["pos"] = self.pos_dt[step][self.batch == idx]
                kwargs["forces"] = self.forces_dt[step][self.batch == idx]
                kwargs["energies"] = self.energies_dt[step, idx]
            else:
                raise ValueError(f"Cannot return step {step}, no pos_dt attribute found in batch.")

        c = Conformer(**kwargs)
        c.to(self.device)
        return c

    @classmethod
    def cat(cls, batches: list["ConformerBatch"]) -> "ConformerBatch":
        """Concatenate multiple ConformerBatch objects into a single batch."""
        batch = cls.from_data_list([conf for cb in batches for conf in cb.to_data_list()])
        batch.__post_init__()
        return batch

    @classmethod
    def from_data_list(cls, data_list: list[Conformer], device: str = "cpu", *args, **kwargs):
        """Create a ConformerBatch from a list of Conformer objects."""
        if len(data_list) == 0:
            raise ValueError("from_data_list received an empty data_list")

        # Gather union of keys and basic graph sizes
        all_keys = set()
        num_nodes_list = []
        num_edges_list = []
        for d in data_list:
            # PyG Data has keys() and len(d)
            all_keys.update(list(d.keys()))
            # num_nodes and num_edges
            n_nodes = int(getattr(d, "num_nodes", 0) or 0)
            if n_nodes == 0 and hasattr(d, "x"):
                # fallback if num_nodes not filled
                try:
                    n_nodes = int(d.x.size(0))
                except Exception:
                    n_nodes = 0
            num_nodes_list.append(n_nodes)

            eidx = getattr(d, "edge_index", None)
            if eidx is not None and hasattr(eidx, "size"):
                n_edges = int(eidx.size(1))
            else:
                n_edges = int(getattr(d, "num_edges", 0) or 0)
            num_edges_list.append(n_edges)

        # Keys that core batching should NOT touch (we will handle below)
        exclude_keys = set()
        reserved = {"edge_index", "ptr", "batch"}  # always left to super()

        def is_tensor(v):
            return isinstance(v, torch.Tensor)

        # Decide exclusion per key using shape/type heuristics
        for key in sorted(all_keys):
            if key in reserved:
                continue

            vals = [getattr(d, key, None) for d in data_list]
            # If all None, exclude and set later as None
            if all(v is None for v in vals):
                exclude_keys.add(key)
                continue

            # Mixed types => exclude
            types = {type(v) for v in vals}
            if len(types - {type(None)}) > 1:
                exclude_keys.add(key)
                continue

            # If any non-tensor (str, dict, list, number) => exclude (we'll listify/stack)
            if not all((v is None) or is_tensor(v) for v in vals):
                exclude_keys.add(key)
                continue

            # All tensors (or None). Heuristics:
            # - Node-level: size(0) == num_nodes => let super() concat along dim=0
            # - Edge-level: size(0) == num_edges => let super() concat along dim=0
            # - Otherwise:
            #     * If shapes all equal (ignoring None) => exclude and stack
            #     * Else => exclude and keep list
            is_node_level = True
            is_edge_level = True

            for i, v in enumerate(vals):
                if v is None:
                    continue

                # Node-level check
                if v.dim() == 0 or v.size(0) != num_nodes_list[i]:
                    is_node_level = False
                # Edge-level check
                if v.dim() == 0 or v.size(0) != num_edges_list[i]:
                    is_edge_level = False

            if not is_node_level and not is_edge_level:
                # Not safely concatenable along dim 0 by super()
                exclude_keys.add(key)
                continue
            # else: let super() handle concatenation

        # Let super() collate the safe attrs, ignoring those we want custom handling
        kwargs = dict(kwargs or {})
        existing_exclude = set(kwargs.get("exclude_keys", []))
        kwargs["exclude_keys"] = sorted(existing_exclude.union(exclude_keys))

        batch = super().from_data_list(data_list, *args, **kwargs)

        # Now attach the excluded keys with robust handling
        for key in sorted(exclude_keys):
            vals = [getattr(d, key, None) for d in data_list]

            # All None -> set None
            if all(v is None for v in vals):
                setattr(batch, key, None)
                continue

            # If all tensors and shapes equal -> stack along new batch dim
            if all((v is not None) and is_tensor(v) for v in vals):
                shapes = [tuple(v.shape) for v in vals]
                if len(set(shapes)) == 1:
                    try:
                        stacked = torch.stack(vals, dim=0)
                        setattr(batch, key, stacked.to(device))
                        continue
                    except Exception:
                        pass  # fallback to list below

            # If all numeric scalars -> make a 1D tensor
            if all(isinstance(v, int | float) for v in vals):
                setattr(batch, key, torch.tensor(vals, device=device))
                continue

            # Fallback: keep as a python list (do not force device)
            setattr(batch, key, vals)

        # Device + finalise
        batch = batch.to(device=device)
        batch.__post_init__()
        return batch

    @classmethod
    def from_rdkit(
        cls, mol: list[Chem.Mol] | Chem.Mol, device: str = "cpu", **kwargs
    ) -> "ConformerBatch":
        """Create a ConformerBatch from a list of RDKit Mol objects.

        Each Mol can have multiple conformers.
        Note : kwargs are applied to all objects in the batch.
        """
        if isinstance(mol, Chem.Mol):
            mol = [mol]

        conformers = []
        for m in mol:
            for conformer in m.GetConformers():
                conformers.append(Conformer.from_rdkit(m, conformer, **kwargs))

        batch = cls.from_data_list(conformers, device=device)
        return batch

    @classmethod
    def from_ase(cls, atoms_list: list[Atoms], device: str = "cpu") -> "ConformerBatch":
        """Create a ConformerBatch from a list of ASE Atoms objects."""
        conformers = [Conformer.from_ase(a) for a in atoms_list]
        return cls.from_data_list(conformers, device=device)

    def to_data_list(self):
        """Convert the batch back to a list of Conformer objects."""
        return [self.conformer(i) for i in range(self.n_conformers)]

    def to_rdkit(self):
        """Convert each conformer in the batch to an RDKit Mol object."""
        return [conformer.to_rdkit() for conformer in self.to_data_list()]

    def to_ase(self):
        """Convert each conformer in the batch to an ASE Atoms object."""
        return [conformer.to_ase() for conformer in self.to_data_list()]


if __name__ == "__main__":
    from ase.build import molecule
    from rdkit.Chem import AllChem

    atoms_list = [molecule("H2O"), molecule("NH3")]
    batch = ConformerBatch.from_ase(atoms_list)
    print(batch)

    confs = [Conformer.from_ase(conf) for conf in atoms_list]
    batch = ConformerBatch.from_data_list(confs)
    print(batch)

    mols_list = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CC")]
    for i, mol in enumerate(mols_list):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mols_list[i] = mol
    new_conf = Chem.Conformer(mols_list[0].GetConformer(0))
    mols_list[0].AddConformer(new_conf)
    batch = ConformerBatch.from_rdkit(mols_list)
    print(batch)

    print(batch.batch)

    print(batch.conformer(0))
