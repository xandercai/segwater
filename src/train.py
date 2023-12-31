from datetime import datetime
from typing import Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_loop(
    epochs: int,
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    acc_fns: Optional[List] = None,
    batch_tfms: Optional[Callable] = None,
    device: Optional[str] = "cuda",
) -> List[float]:
    acc = []
    model.to(device)
    for epoch in range(epochs):
        start = datetime.now()
        accum_loss = 0
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        for batch in train_dl:
            if batch_tfms is not None:
                batch["image"] = batch_tfms(batch["image"])

            X = batch["image"].to(device)
            y = batch["mask"].type(torch.long).to(device)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            pred = model(X)["out"]
            loss = loss_fn(pred, y)

            # BackProp
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # update the accum loss
            accum_loss += float(loss) / len(train_dl)

        # Testing against the validation dataset
        if acc_fns is not None and val_dl is not None:
            # reset the accuracies metrics
            acc = [0.0] * len(acc_fns)

            with torch.no_grad():
                model.eval()
                for batch in val_dl:
                    if batch_tfms is not None:
                        batch["image"] = batch_tfms(batch["image"])

                    X = batch["image"].type(torch.float32).to(device)
                    y = batch["mask"].type(torch.long).to(device)

                    pred = model(X)["out"]

                    for i, acc_fn in enumerate(acc_fns):
                        acc[i] = float(acc[i] + acc_fn(pred, y) / len(val_dl))

        # at the end of the epoch, print the loss, etc.
        print(
            f"Epoch: {epoch:02d}\tLR: {lr:7.5f}\tLoss: {accum_loss:7.5f}\tAccs={[f'{a:5.3f}' for a in acc]}\t"
            f"Time: {(datetime.now() - start)}"
        )

    return acc
