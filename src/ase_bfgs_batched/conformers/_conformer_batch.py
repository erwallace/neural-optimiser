from collections.abc import Iterator
from typing import Any, Union, Optional

from torch_geometric.data import Batch

from ase_bfgs_batched.conformers import Conformer

import numpy as np
from ase import Atoms
from loguru import logger as logging
from rdkit import Chem
import torch

class ConformerBatch(Batch):

    # batch.batch [n_atoms] -> which atoms belongs to which conformer
    # batch.ptr [n_conformers + 1] -> index pointers to start of each conformer in .pos and .atom_types
    # batch.molecule_idxs [n_atoms] -> which molecule each atom belongs to

    atom_type_dtype = torch.int64
    pos_dtype = torch.float32
    molecule_idx_dtype = torch.int64
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   

    def __post_init__(self):
        """Validate attributes."""
        if self.pos.ndim != 2 or self.pos.size(-1) != 3:
            raise ValueError(f"pos must have shape [n_atoms, 3], got {tuple(self.pos.shape)}")
        if self.atom_types.ndim != 1:
            raise ValueError(f"atom_types must be 1-D [n_atoms], got {tuple(self.atom_types.shape)}")
        if self.atom_types.size(0) != self.pos.size(0):
            raise ValueError(
                f"atom_types and pos must have matching n_atoms, "
                f"got {self.atom_types.size(0)} vs {self.pos.size(0)}"
            )
        if self.atom_types.dtype != self.atom_type_dtype:
            raise ValueError(f"atom_types must have dtype {self.atom_type_dtype}, got {self.atom_types.dtype}")
        if self.pos.dtype != self.pos_dtype:
            raise ValueError(f"pos must have dtype {self.pos_dtype}, got {self.pos.dtype}")

    @classmethod
    def from_rdkit(cls, mols: list[Chem.Mol]|Chem.Mol) -> "ConformerBatch":
        if isinstance(mols, Chem.Mol):
            mols = [mols]

        conformers = []

        for mol in mols:
            for conformer in mol.GetConformers():
                conformers.append(Conformer.from_rdkit(conformer))

        return cls.from_data_list(conformers)
    
    def to_rdkit(self) -> list[Chem.Mol]:
        # write list of conformers to list? or do i want to have multiple conformers per molecule?
        pass

if __name__ == "__main__":
    from ase.build import molecule
    confs = [molecule("H2O"), molecule("NH3")]
    confs = [Conformer.from_ase(conf) for conf in confs]
    batch = ConformerBatch.from_data_list(confs)
    print(batch)
