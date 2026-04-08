#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 3C-IVIM PINN at 7T (IR or non-IR) with canonical parameter ranges.

Canonical order:
    [Dpar, Fint, Dint, Fmv, Dmv, S0]

Changes from original:
  - Default --ensemble=20 (paper: 20 instances for repeatability)
  - Warns if ensemble=1
  - Uses sigmoidabs constraint (paper methodology)
"""

import argparse
import json
import time
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch

import IVIMNET.deep as deep
import IVIMNET.simulations as sim
from hyperparams import hyperparams as hp_example


PARAM_RANGES = {
    "Dpar": (0.0001, 0.0015),
    "Fint": (0.005, 0.4),
    "Dint": (0.0015, 0.004),
    "Fmv":  (0.0,   0.05),
    "Dmv":  (0.004, 0.2),
    "S0":   (0.90,  1.10),
}
CONS_ORDER = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def apply_ranges_and_timing(arg, ir, te, tr, ti):
    # Paper (Voorter et al. 2023): sigmoid on D-params, abs on fractions
    arg.net_pars.con = "sigmoidabs"
    arg.net_pars.cons_min = [PARAM_RANGES[k][0] for k in CONS_ORDER]
    arg.net_pars.cons_max = [PARAM_RANGES[k][1] for k in CONS_ORDER]
    arg.net_pars.fitS0 = True
    arg.net_pars.IR = bool(ir)

    # simulation ranges: [Dpar, Fint, Dint, Fmv, Dmv]
    arg.sim.range = (
        [PARAM_RANGES["Dpar"][0], PARAM_RANGES["Fint"][0], PARAM_RANGES["Dint"][0], PARAM_RANGES["Fmv"][0], PARAM_RANGES["Dmv"][0]],
        [PARAM_RANGES["Dpar"][1], PARAM_RANGES["Fint"][1], PARAM_RANGES["Dint"][1], PARAM_RANGES["Fmv"][1], PARAM_RANGES["Dmv"][1]],
    )
    arg.sim.IR = bool(ir)

    arg.rel_times.echotime = float(te)
    arg.rel_times.repetitiontime = float(tr)
    arg.rel_times.inversiontime = float(ti)


def _copy_generated_plots_to_out(out_dir: Path):
    """
    simulations.py saves dependency/accuracy plots under ./results
    deep.py training progress plots under ./plots
    Copy both into out_dir/plots for convenience.
    """
    dst = out_dir / "plots"
    dst.mkdir(parents=True, exist_ok=True)

    for src_dir in (Path.cwd() / "results", Path.cwd() / "plots"):
        if src_dir.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf"):
                for f in src_dir.glob(ext):
                    shutil.copy2(f, dst / f.name)


def main():
    p = argparse.ArgumentParser(description="Train 7T PINN for IVIM (IR/non-IR).")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--ir", required=True, type=str2bool, help="true/false")
    p.add_argument("--te", required=True, type=float, help="TE in ms")
    p.add_argument("--tr", required=True, type=float, help="TR in ms")
    p.add_argument("--ti", required=True, type=float, help="TI in ms (0 for non-IR)")
    p.add_argument("--maxit", default=500, type=int)
    p.add_argument("--lr", default=3e-5, type=float)
    # Paper default: 20 ensemble instances for stable predictions
    p.add_argument("--ensemble", default=20, type=int,
                   help="Number of PINN instances to average (paper: 20)")
    p.add_argument("--repeats", default=1, type=int)
    p.add_argument("--jobs", default=1, type=int)
    p.add_argument(
        "--bvalues", default=None, type=str,
        help="Comma-separated b-values in s/mm² in acquisition order, exactly as stored in "
             "your NIfTI .bval file (e.g. '0,10,10,10,20,20,20,40,40,40,...'). "
             "If omitted, the default scheme from hyperparams.py is used."
    )
    args = p.parse_args()

    if args.ensemble == 1:
        warnings.warn(
            "Running with --ensemble=1. The paper (Voorter et al. 2023) used 20 "
            "ensemble instances to mitigate variable results from repeated training. "
            "Consider using --ensemble=20 for reproducible, stable parameter maps."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    # base hyperparams
    arg = hp_example()
    arg = deep.checkarg(arg)

    # reproducibility
    seed = getattr(arg.train_pars, "seed", 1337)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # training controls
    arg.train_pars.skip_net = False
    arg.train_pars.lr = float(args.lr)
    arg.train_pars.maxit = int(args.maxit)

    # PINN-only
    arg.fit.do_fit = False

    # IMPORTANT: enable plotting
    arg.fig = True

    # simulation/training controls
    arg.sim.n_ensemble = int(args.ensemble)
    arg.sim.repeats = int(args.repeats)
    arg.sim.jobs = int(args.jobs)

    # canonical config + IR mode
    apply_ranges_and_timing(arg, args.ir, args.te, args.tr, args.ti)

    # override b-values if provided on CLI
    if args.bvalues is not None:
        arg.sim.bvalues = np.array([float(b) for b in args.bvalues.split(",")], dtype=float)

    print(
        f"Training PINN | IR={args.ir} | TE={args.te} ms | TR={args.tr} ms | TI={args.ti} ms | "
        f"SNR range=[{arg.sim.snr_min}, {arg.sim.snr_max}] | lr={arg.train_pars.lr} | maxit={arg.train_pars.maxit} | "
        f"ensemble={arg.sim.n_ensemble} | repeats={arg.sim.repeats} | jobs={arg.sim.jobs} | "
        f"bvalues={arg.sim.bvalues.tolist()}"
    )

    t0 = time.time()
    matNN, stability, net = sim.sim(arg.sim.snr_min, arg)
    dt_min = (time.time() - t0) / 60.0
    print(f"Training finished in {dt_min:.2f} min")

    model_name = "PINN_7T_trained_IR.pt" if args.ir else "PINN_7T_trained_nonIR.pt"
    model_path = args.out / model_name
    torch.save(net.state_dict(), model_path)

    np.save(str(args.out / "results_PINN.npy"), np.asarray(matNN))
    np.save(str(args.out / "stability.npy"), np.asarray(stability))

    # copy plots into output folder
    _copy_generated_plots_to_out(args.out)

    meta = {
        "ir": bool(args.ir),
        "te_ms": float(args.te),
        "tr_ms": float(args.tr),
        "ti_ms": float(args.ti),
        "snr_range": [arg.sim.snr_min, arg.sim.snr_max],
        "lr": float(arg.train_pars.lr),
        "maxit": int(arg.train_pars.maxit),
        "ensemble": int(arg.sim.n_ensemble),
        "repeats": int(arg.sim.repeats),
        "jobs": int(arg.sim.jobs),
        "seed": int(seed),
        "constraint": arg.net_pars.con,
        "bvalues": arg.sim.bvalues.tolist(),
        "param_ranges": PARAM_RANGES,
        "cons_order": CONS_ORDER,
        "model_path": str(model_path),
        "plots_dir": str(args.out / "plots"),
    }
    with open(args.out / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved/copies plots to: {args.out / 'plots'}")


if __name__ == "__main__":
    main()