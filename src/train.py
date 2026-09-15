"""Train the LSTM on one fold with one seed, then write test-period predictions.

    python -m src.train --fold 0 --seed 0
    python -m src.train --benchmark 200      # time 200 steps and exit
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .config import LR_SCHEDULE, Cfg
from .data import prepare, sample_batch
from .model import LSTMModel


def nse_loss(pred, obs, sigma, eps=0.1):
    """Basin-normalised squared error (Kratzert et al. 2019 NSE*).

    Dividing by each basin's own training-period flow std stops wet, large basins from
    dominating the gradient. `sigma` is a training-period statistic of the query basin, used
    only to weight the loss -- it never touches the prediction itself.
    """
    m = ~torch.isnan(obs)
    return (((pred[m] - obs[m]) / (sigma[m] + eps)) ** 2).mean()


def steps_per_epoch(cfg: Cfg, data):
    return (data.n_gauged * len(data.train_t)) // cfg.batch_size


def train(cfg: Cfg, data, max_steps=0):
    torch.manual_seed(cfg.seed)
    model = LSTMModel(cfg).to(data.X.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_SCHEDULE[0])
    rng = np.random.default_rng(cfg.seed)
    start = time.time()

    for epoch in range(cfg.epochs):
        if epoch in LR_SCHEDULE:
            for g in opt.param_groups:
                g["lr"] = LR_SCHEDULE[epoch]
        model.train()
        n_steps = min(steps_per_epoch(cfg, data), max_steps) if max_steps else steps_per_epoch(cfg, data)
        total = 0.0
        for _ in range(n_steps):
            x, obs, sigma = sample_batch(data, rng, cfg.batch_size, cfg.seq_len)
            loss = nse_loss(model(x), obs, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1:02d}/{cfg.epochs} | loss {total / n_steps:.4f} "
              f"| lr {opt.param_groups[0]['lr']:.0e} | {(time.time() - start) / 60:.1f} min", flush=True)
    return model


def benchmark(cfg: Cfg, data, n_steps):
    """Time real training steps so the run budget rests on measurement, not estimation."""
    model = LSTMModel(cfg).to(data.X.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(cfg.seed)
    cuda = data.X.device.type == "cuda"
    for i in range(n_steps + 10):
        if i == 10:
            torch.cuda.synchronize() if cuda else None
            start = time.time()
        x, obs, sigma = sample_batch(data, rng, cfg.batch_size, cfg.seq_len)
        loss = nse_loss(model(x), obs, sigma)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
    torch.cuda.synchronize() if cuda else None
    per = (time.time() - start) / n_steps
    spe = steps_per_epoch(cfg, data)
    print(f"{per * 1000:.1f} ms/step, batch {cfg.batch_size} -> {per * spe / 60:.1f} min/epoch, "
          f"{per * spe * cfg.epochs / 3600:.1f} h for {cfg.epochs} epochs")
    print("  single-process timing; concurrent runs share the GPU, so wall clock per run will "
          "be longer than this (total throughput still improves).")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, help="override for smoke tests only")
    p.add_argument("--seq-len", type=int, help="override for smoke tests only")
    p.add_argument("--max-steps", type=int, default=0, help="truncate each epoch (smoke tests)")
    p.add_argument("--benchmark", type=int, default=0, help="time N steps and exit")
    p.add_argument("--save-train-predictions", action="store_true",
                   help="also predict the training period (~60%% more inference; unused here)")
    a = p.parse_args()

    overrides = {k: v for k, v in [("epochs", a.epochs), ("seq_len", a.seq_len)] if v is not None}
    cfg = Cfg(fold=a.fold, seed=a.seed, **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    data = prepare(cfg, device)
    print(f"{cfg.name} | {data.n_gauged} gauged, {len(data.basins) - data.n_gauged} ungauged "
          f"| {len(data.train_t)} train dates, {len(data.test_t)} test dates | {device}", flush=True)

    if a.benchmark:
        benchmark(cfg, data, a.benchmark)
        return

    model = train(cfg, data, a.max_steps)
    out_dir = f"{cfg.out_dir}/{cfg.name}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    json.dump({**vars(cfg), "max_steps": a.max_steps,
               "commit": os.popen("git rev-parse --short HEAD").read().strip()},
              open(f"{out_dir}/config.json", "w"), indent=1)

    from .evaluate import write_predictions
    write_predictions(cfg, model, data, out_dir, a.save_train_predictions)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
