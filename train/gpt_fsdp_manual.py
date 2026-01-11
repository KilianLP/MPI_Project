import argparse
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI

from data.text import get_text_dataloaders
from models import GPT, GPTConfig


def shard_metadata(p: torch.nn.Parameter, world: int) -> Dict:
    """Compute shard sizes/offsets for a flattened parameter."""
    n = p.numel()
    base = n // world
    rem = n % world
    counts = [base + 1 if r < rem else base for r in range(world)]
    displs = [0]
    for i in range(1, world):
        displs.append(displs[i - 1] + counts[i - 1])
    return {"counts": counts, "displs": displs, "numel": n}


def init_shards(params: List[torch.nn.Parameter], rank: int, world: int):
    """Return per-param shard tensors (on CPU) and metadata."""
    metas = []
    shards = []
    for p in params:
        meta = shard_metadata(p, world)
        start = meta["displs"][rank]
        end = start + meta["counts"][rank]
        flat = p.detach().cpu().view(-1)
        shard = flat[start:end].clone()
        metas.append(meta)
        shards.append(shard)
    return metas, shards


def mpi_dtype_from_array(arr: np.ndarray):
    return MPI._typedict[arr.dtype.char]


def allgather_params(params, metas, shards, comm: MPI.Comm, rank: int):
    """Allgather shards to materialize full parameters for compute."""
    for p, meta, shard in zip(params, metas, shards):
        counts = meta["counts"]
        displs = meta["displs"]
        full = np.empty(meta["numel"], dtype=shard.numpy().dtype)
        comm.Allgatherv(
            [shard.numpy(), mpi_dtype_from_array(shard.numpy())],
            [full, counts, displs, mpi_dtype_from_array(shard.numpy())],
        )
        with torch.no_grad():
            p.data.copy_(torch.from_numpy(full).view_as(p))


def reduce_scatter_grads(params, metas, shards, lr: float, comm: MPI.Comm, rank: int):
    """
    Approximate reduce-scatter: Allreduce full grad, then apply local shard slice.
    This keeps logic simple while matching aggregated shard updates.
    """
    world = comm.Get_size()
    for p, meta, shard in zip(params, metas, shards):
        if p.grad is None:
            continue
        grad = p.grad.detach().cpu().contiguous().view(-1)
        buf = grad.numpy()
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        buf /= world
        start = meta["displs"][rank]
        end = start + meta["counts"][rank]
        shard -= lr * torch.from_numpy(buf[start:end])
        p.grad = None  # free grad


def reduce_scalar(val: float, comm: MPI.Comm):
    buf = np.array([val], dtype="float64")
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    return buf.item() / comm.Get_size()


def save_checkpoint(model, metas, shards, optimizer_state, epoch, step, config, path, rank, comm):
    if rank != 0:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "step": step,
        "config": asdict(config),
    }
    torch.save(payload, path)
    print(f"[gpt-fsdp-mpi] saved checkpoint to {path} (epoch {epoch}, step {step})")


def evaluate(model, val_loader, comm, rank):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to("cpu")
            y = y.to("cpu")
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += y.numel()
    metrics = np.array([total_loss, total_tokens], dtype="float64")
    comm.Allreduce(MPI.IN_PLACE, metrics, op=MPI.SUM)
    if metrics[1] == 0:
        return
    loss_avg = metrics[0] / metrics[1]
    ppl = np.exp(loss_avg) if loss_avg < 20 else float("inf")
    if rank == 0:
        print(f"[gpt-fsdp-mpi] eval_loss={loss_avg:.4f} eval_ppl={ppl:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Manual FSDP-like GPT training with mpi4py on tinyshakespeare.")
    parser.add_argument("--file", type=str, default=os.path.join("data", "tinyshakespeare.txt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str, default=os.path.join("checkpoints", "gpt_fsdp_manual.pt"))
    parser.add_argument("--save-every-steps", type=int, default=0, help="Save every N steps (0 disables intra-epoch saves).")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    if not os.path.isfile(args.file):
        if rank == 0:
            raise FileNotFoundError(f"Data file not found: {args.file}")
        return

    torch.set_num_threads(1)
    if rank == 0:
        print(f"Running FSDP GPT training (MPI) on CPU, world={world}")

    train_loader, val_loader, vocab_size = get_text_dataloaders(
        file_path=args.file,
        block_size=args.block_size,
        batch_size=args.batch_size,
        rank=rank,
        world=world,
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    torch.manual_seed(42 + rank)
    model = GPT(config).to("cpu")
    params = list(model.parameters())
    metas, shards = init_shards(params, rank, world)

    # Initial materialization for first forward
    allgather_params(params, metas, shards, comm, rank)

    for epoch in range(args.epochs):
        model.train()
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        for step, (x, y) in enumerate(train_loader):
            # Materialize params for this step
            allgather_params(params, metas, shards, comm, rank)

            x = x.to("cpu")
            y = y.to("cpu")

            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            reduce_scatter_grads(params, metas, shards, args.lr, comm, rank)

            if step % 100 == 0:
                loss_avg = reduce_scalar(loss.item(), comm)
                if rank == 0:
                    print(f"[gpt-fsdp-mpi] epoch={epoch} step={step} loss_avg={loss_avg:.4f}")

            if args.save_every_steps > 0 and (step + 1) % args.save_every_steps == 0:
                allgather_params(params, metas, shards, comm, rank)
                save_checkpoint(model, metas, shards, {}, epoch, step, config, args.save_path, rank, comm)

        allgather_params(params, metas, shards, comm, rank)
        evaluate(model, val_loader, comm, rank)
        save_checkpoint(model, metas, shards, {}, epoch, -1, config, args.save_path, rank, comm)

    comm.Barrier()
    if rank == 0:
        print("[gpt-fsdp-mpi] done.")


if __name__ == "__main__":
    main()
