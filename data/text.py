import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SimpleCharTokenizer:
    def __init__(self, text: str):
        vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class TextSeqDataset(Dataset):
    """Character-level language modeling dataset with rank-based striding."""

    def __init__(self, ids: torch.Tensor, block_size: int, rank: int, world: int):
        if ids.ndim != 1:
            raise ValueError("ids tensor must be 1D")
        self.ids = ids
        self.block_size = block_size
        self.rank = rank
        self.world = world
        self.total_positions = max(0, ids.numel() - block_size)

    def __len__(self):
        if self.total_positions <= 0:
            return 0
        return max(0, (self.total_positions - self.rank + self.world - 1) // self.world)

    def __getitem__(self, idx):
        start = self.rank + idx * self.world
        if start >= self.total_positions:
            raise IndexError
        x = self.ids[start : start + self.block_size].long()
        y = self.ids[start + 1 : start + 1 + self.block_size].long()
        return x, y


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_ids(ids: List[int], train_frac: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(ids)
    n_train = int(n * train_frac)
    train_ids = torch.tensor(ids[:n_train], dtype=torch.long)
    val_ids = torch.tensor(ids[n_train:], dtype=torch.long)
    return train_ids, val_ids


def get_text_dataloaders(
    file_path: str,
    block_size: int,
    batch_size: int,
    rank: int,
    world: int,
    num_workers: int = 0,
    train_frac: float = 0.9,
    cache_dir: Optional[str] = None,
):
    """
    Build DataLoaders for tinyshakespeare-style character LM.
    Shards samples across ranks by striding start positions.
    """
    text = load_text(file_path)
    tokenizer = SimpleCharTokenizer(text)
    ids = tokenizer.encode(text)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    train_ids, val_ids = split_ids(ids, train_frac=train_frac)

    train_ds = TextSeqDataset(train_ids, block_size=block_size, rank=rank, world=world)
    val_ds = TextSeqDataset(val_ids, block_size=block_size, rank=rank, world=world)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, tokenizer.vocab_size
