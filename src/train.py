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
from .data import layout, prepare, sample_batch, step_inputs
from .model import PUBModel


def nse_loss(pred, obs, sigma, eps=0.1):
    """Basin-normalised squared error (Kratzert et al. 2019 NSE*).

    Dividing by each basin's own training-period flow std stops wet, large basins from
    dominating the gradient. `sigma` is a training-period statistic of the *query* basin,
    used only to weight the loss -- it never touches the prediction itself.
    """
    m = ~torch.isnan(obs)
    return (((pred[m] - obs[m]) / (sigma[m] + eps)) ** 2).mean()


def _opt_step(model, opt, cfg, pred, obs, sigma):
    loss = nse_loss(pred, obs, sigma)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
    opt.step()
    return loss.detach()


def steps_per_epoch(cfg: Cfg, data):
    """Literature-batched (`vanilla_lit`) only; the other architectures step once per date."""
    return (data.n_gauged * len(data.train_t)) // cfg.batch_size


def train(cfg: Cfg, data, device, max_steps=0):
    """Two training loops share this function: the other architectures batch every gauged basin
    at one shared date (`step_inputs`); `vanilla_lit` (`cfg.batch_size` set) instead draws fixed-
    size batches of independently sampled (basin, date-window) pairs (`sample_batch`) -- the
    literature-standard scheme. Both share the optimizer step (`_opt_step`) and LR schedule.
    """
    torch.manual_seed(cfg.seed)
    model = PUBModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_SCHEDULE[0])
    rng = np.random.default_rng(cfg.seed)
    start = time.time()

    if cfg.batch_size is None:
        n_enc, n_query, n_donor = layout(cfg, data, training=True)
        sigma_all = data.sigma[:n_query]

    for epoch in range(cfg.epochs):
        if epoch in LR_SCHEDULE:
            for g in opt.param_groups:
                g["lr"] = LR_SCHEDULE[epoch]
        model.train()
        total = torch.zeros((), device=device)

        if cfg.batch_size is None:
            dates = rng.permutation(data.train_t)
            if max_steps:
                dates = dates[:max_steps]
            for t in dates:
                x, dz, valid = step_inputs(cfg, data, int(t), n_enc, n_donor)
                pred, _ = model(x, n_query, n_donor, dz, valid)
                total += _opt_step(model, opt, cfg, pred, data.Q[:n_query, t], sigma_all)
            n_steps = len(dates)
        else:
            n_steps = steps_per_epoch(cfg, data)
            if max_steps:
                n_steps = min(n_steps, max_steps)
            for _ in range(n_steps):
                x, obs, sigma = sample_batch(data, rng, data.n_gauged, data.train_t,
                                             cfg.batch_size, cfg.seq_len)
                pred, _ = model(x, x.shape[0])
                total += _opt_step(model, opt, cfg, pred, obs, sigma)

        print(f"epoch {epoch + 1:02d}/{cfg.epochs} | loss {total.item() / n_steps:.4f} "
              f"| lr {opt.param_groups[0]['lr']:.0e} | {(time.time() - start) / 60:.1f} min",
              flush=True)
    return model


def benchmark(cfg: Cfg, data, device, n_steps):
    """Time real training steps so the run budget rests on measurement, not estimation."""
    model = PUBModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(cfg.seed)

    if cfg.batch_size is None:
        n_enc, n_query, n_donor = layout(cfg, data, training=True)
        dates = data.train_t[:n_steps + 10]
        def one_step(i):
            t = dates[i]
            x, dz, valid = step_inputs(cfg, data, int(t), n_enc, n_donor)
            pred, _ = model(x, n_query, n_donor, dz, valid)
            return nse_loss(pred, data.Q[:n_query, t], data.sigma[:n_query])
        per_batch = n_enc
        epoch_steps = len(data.train_t)
    else:
        def one_step(i):
            x, obs, sigma = sample_batch(data, rng, data.n_gauged, data.train_t,
                                         cfg.batch_size, cfg.seq_len)
            pred, _ = model(x, x.shape[0])
            return nse_loss(pred, obs, sigma)
        per_batch = cfg.batch_size
        epoch_steps = steps_per_epoch(cfg, data)

    for i in range(n_steps + 10):
        if i == 10:  # discard warm-up
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.time()
        loss = one_step(i)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
    torch.cuda.synchronize() if device.type == "cuda" else None
    per_step = (time.time() - start) / n_steps
    hours = per_step * epoch_steps * cfg.epochs / 3600
    print(f"{cfg.arch}: {per_step * 1000:.1f} ms/step, {per_batch} basins/batch "
          f"-> {per_step * epoch_steps / 60:.1f} min/epoch, {hours:.1f} h for {cfg.epochs} epochs")
    print("  single-process timing; concurrent runs share the GPU, so wall clock per run will "
          "be longer than this (total throughput still improves).")


def save_attention(cfg: Cfg, model, data, device, out_dir, n_dates=20):
    """Dump attention for a few evenly spaced test dates, so a null result is diagnosable."""
    n_enc, n_query, n_donor = layout(cfg, data, training=False)
    dates = data.test_t[:: max(1, len(data.test_t) // n_dates)][:n_dates]
    model.eval()
    mats = []
    with torch.no_grad():
        for t in dates:
            x, dz, valid = step_inputs(cfg, data, int(t), n_enc, n_donor)
            _, attn = model(x, n_query, n_donor, dz, valid, want_attn=True)
            mats.append(attn.half().cpu().numpy())
    np.savez_compressed(f"{out_dir}/attention.npz", attn=np.stack(mats), dates=dates,
                        n_gauged=data.n_gauged)


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
    if a.arch == "vanilla_lit":
        # Literature-faithful (Kratzert et al. 2019, WRR -- the paper BENCHMARKS.md's external
        # reference comes from): forget-gate bias 5, not this project's frozen 3. Batch size is
        # the NeuralHydrology default (256), not held to the other architectures' frozen set --
        # see CLAUDE.md.
        overrides = {"forget_bias": 5.0, "batch_size": 256, **overrides}
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
