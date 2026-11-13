"""Train a small MLP on a synthetic classification task and log the results to IVAN."""
import hydra
import ivan
import torch
from omegaconf import DictConfig
from torch import nn


def make_data(cfg: DictConfig, gen: torch.Generator):
    """Synthetic 2-class task: a fixed random nonlinear function of the inputs, plus label noise."""
    n, d = cfg.n_samples, cfg.n_features
    x = torch.randn(n, d, generator=gen)
    w1 = torch.randn(d, 32, generator=torch.Generator().manual_seed(1234))  # task is fixed across seeds
    w2 = torch.randn(32, generator=torch.Generator().manual_seed(1235))
    logits = torch.tanh(x @ w1) @ w2 / 32**0.5
    y = (logits + cfg.noise * torch.randn(n, generator=gen) > 0).long()
    n_val = int(n * cfg.val_frac)
    return (x[n_val:], y[n_val:]), (x[:n_val], y[:n_val])


def make_model(cfg: DictConfig, n_in: int) -> nn.Module:
    layers, width = [], n_in
    for _ in range(cfg.layers):
        layers += [nn.Linear(width, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout)]
        width = cfg.hidden
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, x, y, loss_fn):
    model.eval()
    out = model(x)
    return loss_fn(out, y).item(), (out.argmax(1) == y).float().mean().item()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    (xt, yt), (xv, yv) = make_data(cfg.data, torch.Generator().manual_seed(cfg.seed))
    xt, yt, xv, yv = (t.to(device) for t in (xt, yt, xv, yv))

    model = make_model(cfg.model, xt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    loss_fn = nn.CrossEntropyLoss()
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")

    best = 0.0
    for epoch in range(cfg.train.epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        total = 0.0
        for i in range(0, len(xt), cfg.train.batch_size):
            idx = perm[i:i + cfg.train.batch_size]
            loss = loss_fn(model(xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_loss, val_acc = evaluate(model, xv, yv, loss_fn)
        best = max(best, val_acc)
        ivan.log_step({"train_loss": total / len(xt), "val_loss": val_loss, "val_acc": val_acc,
                       "lr": sched.get_last_lr()[0]}, step=epoch)
        print(f"epoch {epoch:3d}  train_loss {total / len(xt):.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

    ivan.log_summary({"val_acc": best, "final_val_acc": val_acc, "final_val_loss": val_loss})
    print(f"best val_acc {best:.4f}")
    return best


if __name__ == "__main__":
    main()
