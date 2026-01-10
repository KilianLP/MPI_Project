import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI

from data.mnist import get_mnist_dataloaders
from models import MnistCNN


def average_gradients(model: nn.Module, comm: MPI.Comm):
    """In-place average of gradients across all ranks."""
    world = comm.Get_size()
    for p in model.parameters():
        if p.grad is None:
            continue
        grad = p.grad.detach()
        needs_copy_back = False
        if grad.is_cuda:
            raise RuntimeError("CPU-only example; use torch.distributed with NCCL for GPU.")
        if not grad.is_contiguous():
            grad = grad.contiguous()
            needs_copy_back = True

        buf = grad.numpy()
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        grad.mul_(1.0 / world)

        if needs_copy_back:
            p.grad.copy_(grad)


def broadcast_parameters(model: nn.Module, comm: MPI.Comm, root: int = 0):
    """Ensure all ranks start from the same parameters."""
    for p in model.parameters():
        tensor = p.detach()
        if tensor.is_cuda:
            raise RuntimeError("CPU-only example; use torch.distributed with NCCL for GPU.")
        buf = tensor.numpy()
        comm.Bcast(buf, root=root)


def reduce_scalar(value: float, comm: MPI.Comm):
    """Average a Python float across ranks using a fast buffer Allreduce."""
    buf = np.array(value, dtype="float64")
    comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
    return buf.item() / comm.Get_size()


def evaluate(model, loader, loss_fn, device, comm):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            logits = model(data)
            loss = loss_fn(logits, target)

            total_loss += loss.item() * data.size(0)
            total_correct += (logits.argmax(dim=1) == target).sum().item()
            total_samples += data.size(0)

    metrics = np.array([total_loss, total_correct, total_samples], dtype="float64")
    comm.Allreduce(MPI.IN_PLACE, metrics, op=MPI.SUM)

    if metrics[2] == 0:
        return 0.0, 0.0

    global_loss = metrics[0] / metrics[2]
    global_acc = metrics[1] / metrics[2]
    return global_loss, global_acc


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()

    torch.set_num_threads(1)
    device = torch.device("cpu")

    batch_size = 64
    epochs = 2
    lr = 0.01
    data_root = os.path.join("data", "mnist")

    # Use a reliable mirror to avoid cert issues on some platforms.
    mirror = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    if rank == 0:
        os.makedirs(data_root, exist_ok=True)
        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=batch_size,
            data_root=data_root,
            rank=rank,
            world=world,
            download=True,
            mirror_url=mirror,
        )
        comm.Barrier()
    else:
        comm.Barrier()
        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=batch_size,
            data_root=data_root,
            rank=rank,
            world=world,
            download=False,
            mirror_url=mirror,
        )

    model = MnistCNN().to(device)
    broadcast_parameters(model, comm)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = loss_fn(logits, target)
            loss.backward()

            average_gradients(model, comm)
            optimizer.step()

            if step % 100 == 0:
                loss_avg = reduce_scalar(loss.item(), comm)
                if rank == 0:
                    print(f"epoch={epoch} step={step} loss_avg={loss_avg:.4f}")

        val_loss, val_acc = evaluate(model, test_loader, loss_fn, device, comm)
        if rank == 0:
            print(f"epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

    # Check parameters match across ranks
    with torch.no_grad():
        flat_params = torch.cat([p.view(-1).cpu() for p in model.parameters()])
        checksum = float(flat_params.abs().sum().item())
        checksums = comm.gather(checksum, root=0)
        if rank == 0:
            print("weight checksums per rank:", checksums)


if __name__ == "__main__":
    main()
