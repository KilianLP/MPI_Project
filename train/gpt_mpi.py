import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mpi4py import MPI

from data.text import get_text_dataloaders
from models import GPT, GPTConfig


def average_gradients(model: torch.nn.Module, comm: MPI.Comm):
    world = comm.Get_size()
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        needs_copy_back = False
        if grad.is_cuda:
            raise RuntimeError("CPU-only example; use torch.distributed for GPU.")
        if not grad.is_contiguous():
            grad = grad.contiguous()
            needs_copy_back = True
        buf = grad.numpy()
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        grad.mul_(1.0 / world)
        if needs_copy_back:
            p.grad.copy_(grad)


def broadcast_parameters(model: torch.nn.Module, comm: MPI.Comm, root: int = 0):
    for p in model.parameters():
        tensor = p.detach()
        if tensor.is_cuda:
            raise RuntimeError("CPU-only example.")
        buf = tensor.numpy()
        comm.Bcast(buf, root=root)


def reduce_scalar(value: float, comm: MPI.Comm):
    buf = np.array(value, dtype="float64")
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    return buf.item() / comm.Get_size()


@torch.no_grad()
def evaluate(model, loader, device, comm):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()

    metrics = np.array([total_loss, total_tokens], dtype="float64")
    comm.Allreduce(MPI.IN_PLACE, metrics, op=MPI.SUM)
    if metrics[1] == 0:
        return 0.0
    return metrics[0] / metrics[1]


def save_checkpoint(model, optimizer, epoch, step, config, path, rank):
    if rank != 0:
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": asdict(config),
    }
    torch.save(payload, path)
    print(f"[gpt-mpi] saved checkpoint to {path} (epoch {epoch}, step {step})")


def main():
    parser = argparse.ArgumentParser(description="TinyShakespeare GPT training with MPI data parallelism.")
    parser.add_argument("--file", type=str, default=os.path.join("data", "tinyshakespeare.txt"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-path", type=str, default=os.path.join("checkpoints", "gpt_checkpoint.pt"))
    parser.add_argument("--save-every-steps", type=int, default=0, help="Save every N steps (0 disables intra-epoch saves).")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    torch.set_num_threads(1)
    device = torch.device("cpu")

    if not os.path.isfile(args.file):
        if rank == 0:
            raise FileNotFoundError(f"Data file not found: {args.file}")
        else:
            return

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
    model = GPT(config).to(device)

    broadcast_parameters(model, comm)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            average_gradients(model, comm)
            optimizer.step()

            if step % 100 == 0:
                loss_avg = reduce_scalar(loss.item(), comm)
                if rank == 0:
                    print(f"[gpt-mpi] epoch={epoch} step={step} loss_avg={loss_avg:.4f}")

            if args.save_every_steps > 0 and (step + 1) % args.save_every_steps == 0:
                save_checkpoint(model, optimizer, epoch, step, config, args.save_path, rank)

        val_loss = evaluate(model, val_loader, device, comm)
        if rank == 0:
            ppl = np.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"[gpt-mpi] epoch={epoch} val_loss={val_loss:.4f} val_ppl={ppl:.2f}")

        save_checkpoint(model, optimizer, epoch, step, config, args.save_path, rank)

    comm.Barrier()
    if rank == 0:
        print("[gpt-mpi] done.")


if __name__ == "__main__":
    main()
