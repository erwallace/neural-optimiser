from typing import Any, Union, Optional

from torch_geometric.data import Batch

from ase_bfgs_batched.conformers import Conformer

import numpy as np
from ase import Atoms
from loguru import logger
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
        self.__post_init__()   

    def __post_init__(self):
        """Validate attributes."""
        if getattr(self, "atom_types", None) is None or getattr(self, "pos", None) is None:
            return  # placeholder instance during Batch.from_data_list
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
        
        # Not set when initialised using from_data_list()
        if not hasattr(self, 'molecule_idxs'):
            self.molecule_idxs = self.batch.clone()
        

    @classmethod
    def from_rdkit(cls, mols: list[Chem.Mol]|Chem.Mol) -> "ConformerBatch":
        if isinstance(mols, Chem.Mol):
            mols = [mols]

        conformers = []
        molecule_idxs = []
        for i, mol in enumerate(mols):
            for conformer in mol.GetConformers():
                conformers.append(Conformer.from_rdkit(mol, conformer))
                molecule_idxs += [i] * mol.GetNumAtoms()

        batch =  cls.from_data_list(conformers)
        batch.molecule_idxs = torch.tensor(molecule_idxs, dtype=cls.molecule_idx_dtype)
        return batch
    
    @classmethod 
    def from_ase(cls, atoms_list: list[Atoms]) -> "ConformerBatch":
        conformers = [Conformer.from_ase(a) for a in atoms_list]
        return cls.from_data_list(conformers)
    
    def to_rdkit(self) -> list[Chem.Mol]:
        # write list of conformers to list? or do i want to have multiple conformers per molecule?
        raise NotImplementedError("ConformerBatch.to_rdkit() not implemented yet.")

if __name__ == "__main__":
    from rdkit.Chem import AllChem
    from ase.build import molecule
    
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
    print(batch.molecule_idxs)
