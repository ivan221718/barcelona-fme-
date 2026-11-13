# barcelona-fme

A small MLP pipeline (PyTorch + Hydra) on a synthetic 2-class task, wired for IVAN.

| Path | What |
|---|---|
| `configs/config.yaml` | all hyperparameters, plus the `sweep:` section IVAN reads |
| `src/train.py` | data, model, training loop; logs `val_acc` etc. with `ivan.log_step` / `ivan.log_summary` |
| `src/ivan/runs/overview.py` | the run page on the IVAN dashboard (loss + accuracy curves) |

## Local

```sh
uv sync
uv run python -m src.train                                # defaults
uv run python -m src.train model.hidden=256 train.lr=3e-3 # any override
```

## On IVAN

```sh
ivan run src.train train.epochs=2 --force --name smoke --hypothesis "it starts"   # smoke test
ivan run src.train train.lr=1e-3,3e-3,1e-2 --name lr-grid --hypothesis "..."       # quick grid, one run per value
ivan sweep src.train                                                               # the sweep: section of config.yaml
ivan status / ivan logs mlp-sweep/3 / ivan experiment list / ivan open
```

Change the sweep by editing `sweep:` in `configs/config.yaml` (method `random`, `grid` or `smac3`; `metric` is `val_acc`), or on the command line: `ivan sweep src.train sweep.method=smac3 sweep.run_cap=30`.
