from collections.abc import Sequence
from functools import partial

from neural_optimiser.conformers import Conformer, ConformerBatch
from torch.utils.data import DataLoader, Dataset


class ConformerDataset(Dataset):
    """Dataset of individual Conformer objects."""

    def __init__(self, conformers: Sequence[Conformer]):
        self.conformers = list(conformers)

    def __len__(self) -> int:
        return len(self.conformers)

    def __getitem__(self, idx: int) -> Conformer:
        return self.conformers[idx]


def collate_conformers(conformers: list[Conformer], device: str = "cpu") -> ConformerBatch:
    """Collate a list of Conformer into a ConformerBatch."""
    return ConformerBatch.from_data_list(conformers, device=device)


class ConformerDataLoader(DataLoader):
    """DataLoader that returns ConformerBatch batches from ConformerDataset."""

    def __init__(
        self,
        dataset: ConformerDataset,
        batch_size: int = 1,
        device: str = "cpu",
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(collate_conformers, device=device),
            **kwargs,
        )
