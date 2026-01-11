"""
Manual, CPU-only FSDP-style training using mpi4py on tinyshakespeare:
- Parameters assigned to owner ranks (balanced by numel).
- All ranks run full forward/backward; grads allreduced.
- Only owners apply optimizer updates; owners broadcast updated params.

This mirrors the transformer data-parallel setup but with explicit sharded
updates over MPI. Not memory-optimal (full params on each rank for compute),
but fully avoids torch.distributed backends.
"""

import argparse
import os
from dataclasses import asdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

from data.text import get_text_dataloaders
from models import GPT, GPTConfig


def assign_param_owners(params: List[nn.Parameter], world: int) -> List[int]:
    loads = [0] * world
    owners = []
    for p in params:
        r = min(range(world), key=lambda i: loads[i])
        owners.append(r)
        loads[r] += p.numel()
    return owners


def average_gradients(params: List[nn.Parameter], comm: MPI.Comm):
    world = comm.Get_size()
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        needs_copy_back = False
        if grad.is_cuda:
            raise RuntimeError("CPU-only example; use torch.distributed+NCCL for GPU.")
        if not grad.is_contiguous():
            grad = grad.contiguous()
            needs_copy_back = True
        buf = grad.numpy()
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        grad.mul_(1.0 / world)
        if needs_copy_back:
            p.grad.copy_(grad)


def broadcast_parameters(params: List[nn.Parameter], owners: List[int], comm: MPI.Comm):
    for p, owner in zip(params, owners):
        tensor = p.detach()
        if tensor.is_cuda:
            raise RuntimeError("CPU-only example.")
        buf = tensor.numpy()
        comm.Bcast(buf, root=owner)


def reduce_scalars(vals: np.ndarray, comm: MPI.Comm):
    comm.Allreduce(MPI_IN_PLACE := MPI.IN_PLACE, vals, op=MPI.SUM)  # noqa: F841
    return vals


def train_one_epoch(model, train_loader, loss_fn, lr, params, owners, comm, rank, epoch, save_every_steps, save_path, config, optimizer_state):
    model.train()
    if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    for step, (x, y) in enumerate(train_loader):
        x = x.to("cpu")
        y = y.to("cpu")

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        average_gradients(params, comm)
        # Simple SGD update; optimizer_state unused placeholder to mirror checkpoint payload.
        for p, owner in zip(params, owners):
            if p.grad is None:
                continue
            if rank == owner:
                p.data.add_(p.grad, alpha=-lr)

        broadcast_parameters(params, owners, comm)

        if step % 100 == 0:
            loss_avg = reduce_scalars(np.array([loss.item()], dtype="float64"), comm)[0] / comm.Get_size()
            if rank == 0:
                print(f"[gpt-fsdp-mpi] epoch={epoch} step={step} loss_avg={loss_avg:.4f}")

        if save_every_steps > 0 and (step + 1) % save_every_steps == 0:
            save_checkpoint(model, optimizer_state, epoch, step, config, save_path, rank)


@torch.no_grad()
def evaluate(model, loader, loss_fn, comm, rank):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x = x.to("cpu")
        y = y.to("cpu")
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

    metrics = np.array([total_loss, total_tokens], dtype="float64")
    reduce_scalars(metrics, comm)
    if rank == 0 and metrics[1] > 0:
        loss_avg = metrics[0] / metrics[1]
        ppl = np.exp(loss_avg) if loss_avg < 20 else float("inf")
        print(f"[gpt-fsdp-mpi] eval_loss={loss_avg:.4f} eval_ppl={ppl:.2f}")


def save_checkpoint(model, optimizer_state, epoch, step, config, path, rank):
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


def main():
    parser = argparse.ArgumentParser(description="Manual FSDP-style GPT training with mpi4py on tinyshakespeare.")
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
        else:
            return

    torch.set_num_threads(1)
    device = torch.device("cpu")
    if rank == 0:
        print(f"Running manual FSDP-style GPT training on device={device}, world={world}")

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
    model = GPT(config).to(device)
    params = list(model.parameters())
    owners = assign_param_owners(params, world)

    broadcast_parameters(params, owners, comm)

    loss_fn = F.cross_entropy
    optimizer_state = {}  # placeholder for symmetry with saved payload

    for epoch in range(args.epochs):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            lr=args.lr,
            params=params,
            owners=owners,
            comm=comm,
            rank=rank,
            epoch=epoch,
            save_every_steps=args.save_every_steps,
            save_path=args.save_path,
            config=config,
            optimizer_state=optimizer_state,
        )
        evaluate(model, val_loader, loss_fn, comm, rank)
        save_checkpoint(model, optimizer_state, epoch, -1, config, args.save_path, rank)

    comm.Barrier()
    if rank == 0:
        print("[gpt-fsdp-mpi] done.")


if __name__ == "__main__":
    main()
