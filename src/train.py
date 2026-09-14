"""Train one architecture on one fold with one seed, then write test-period predictions.

    python -m src.train --arch spatial --fold 0 --seed 0
    python -m src.train --arch donor --benchmark 200      # time 200 steps and exit
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .config import ARCHS, LR_SCHEDULE, Cfg
from .data import layout, prepare, step_inputs
from .model import PUBModel


def nse_loss(pred, obs, sigma, eps=0.1):
    """Basin-normalised squared error (Kratzert et al. 2019 NSE*).

    Dividing by each basin's own training-period flow std stops wet, large basins from
    dominating the gradient. `sigma` is a training-period statistic of the *query* basin,
    used only to weight the loss -- it never touches the prediction itself.
    """
    m = ~torch.isnan(obs)
    return (((pred[m] - obs[m]) / (sigma[m] + eps)) ** 2).mean()


def train(cfg: Cfg, data, device, max_steps=0):
    torch.manual_seed(cfg.seed)
    model = PUBModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_SCHEDULE[0])
    n_enc, n_query, n_donor = layout(cfg, data, training=True)
    sigma = data.sigma[:n_query]
    rng = np.random.default_rng(cfg.seed)
    start = time.time()

    for epoch in range(cfg.epochs):
        if epoch in LR_SCHEDULE:
            for g in opt.param_groups:
                g["lr"] = LR_SCHEDULE[epoch]
        model.train()
        dates = rng.permutation(data.train_t)
        if max_steps:
            dates = dates[:max_steps]
        total = torch.zeros((), device=device)
        for t in dates:
            x, dz, valid, dist = step_inputs(cfg, data, int(t), n_enc, n_donor, n_query)
            pred, _, _ = model(x, n_query, n_donor, dz, valid, dist)
            loss = nse_loss(pred, data.Q[:n_query, t], sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            total += loss.detach()
        print(f"epoch {epoch + 1:02d}/{cfg.epochs} | loss {total.item() / len(dates):.4f} "
              f"| lr {opt.param_groups[0]['lr']:.0e} | {(time.time() - start) / 60:.1f} min",
              flush=True)
    return model


def benchmark(cfg: Cfg, data, device, n_steps):
    """Time real training steps so the run budget rests on measurement, not estimation."""
    model = PUBModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_enc, n_query, n_donor = layout(cfg, data, training=True)
    dates = data.train_t[:n_steps + 10]
    for i, t in enumerate(dates):
        if i == 10:  # discard warm-up
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.time()
        x, dz, valid, dist = step_inputs(cfg, data, int(t), n_enc, n_donor, n_query)
        pred, _, _ = model(x, n_query, n_donor, dz, valid, dist)
        loss = nse_loss(pred, data.Q[:n_query, t], data.sigma[:n_query])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    per_step = (time.time() - start) / n_steps
    hours = per_step * len(data.train_t) * cfg.epochs / 3600
    print(f"{cfg.arch}: {per_step * 1000:.1f} ms/step, {n_enc} basins encoded "
          f"-> {per_step * len(data.train_t) / 60:.1f} min/epoch, {hours:.1f} h for {cfg.epochs} epochs")
    print("  single-process timing; concurrent runs share the GPU, so wall clock per run will "
          "be longer than this (total throughput still improves).")


def save_attention(cfg: Cfg, model, data, device, out_dir, n_dates=20):
    """Dump attention for a few evenly spaced test dates, so a null result is diagnosable."""
    n_enc, n_query, n_donor = layout(cfg, data, training=False)
    dates = data.test_t[:: max(1, len(data.test_t) // n_dates)][:n_dates]
    model.eval()
    mats, gates = [], []
    with torch.no_grad():
        for t in dates:
            x, dz, valid, dist = step_inputs(cfg, data, int(t), n_enc, n_donor, n_query)
            _, attn, gate = model(x, n_query, n_donor, dz, valid, dist, want_attn=True)
            mats.append(attn.half().cpu().numpy())
            if gate is not None:
                gates.append(gate.squeeze(-1).half().cpu().numpy())
    extra = {"gate": np.stack(gates), "dist_w": model.dist_w.detach().item()} if cfg.gated else {}
    np.savez_compressed(f"{out_dir}/attention.npz", attn=np.stack(mats), dates=dates,
                        n_gauged=data.n_gauged, **extra)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, default="vanilla")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, help="override for smoke tests")
    p.add_argument("--seq-len", type=int, help="override for smoke tests")
    p.add_argument("--max-steps", type=int, default=0, help="truncate each epoch (smoke tests)")
    p.add_argument("--benchmark", type=int, default=0, help="time N steps and exit")
    p.add_argument("--pred-stride", type=int, default=1, help="subsample eval dates (smoke tests)")
    a = p.parse_args()

    overrides = {k: v for k, v in [("epochs", a.epochs), ("seq_len", a.seq_len)]
                 if v is not None}
    cfg = Cfg(arch=a.arch, fold=a.fold, seed=a.seed, **overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare(cfg, device)
    print(f"{cfg.name} | {data.n_gauged} gauged, {len(data.basins) - data.n_gauged} ungauged "
          f"| {len(data.train_t)} train dates, {len(data.test_t)} test dates | {device}")

    if a.benchmark:
        benchmark(cfg, data, device, a.benchmark)
        return

    model = train(cfg, data, device, a.max_steps)

    out_dir = f"{cfg.out_dir}/{cfg.name}"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    json.dump({**vars(cfg), "max_steps": a.max_steps, "pred_stride": a.pred_stride,
               "commit": os.popen("git rev-parse --short HEAD").read().strip()},
              open(f"{out_dir}/config.json", "w"), indent=1)

    from .evaluate import write_predictions
    write_predictions(cfg, model, data, out_dir, a.pred_stride)
    if cfg.spatial:
        save_attention(cfg, model, data, device, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
