#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final 7T IVIM runner (PINN + optional NNLS), supports IR and non-IR from CLI.
Uses canonical ranges/order:

SAVE ORDER:
    [Dpar, Fint, Dint, Fmv, Dmv, S0]
"""

import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import IVIMNET.deep as deep
import IVIMNET.fitting_algorithms as fit
from hyperparams import hyperparams as hp_example


# predict_IVIM output order from deep.py
LIB_OUT_ORDER = ["Dpar", "Fmv", "Dmv", "Dint", "Fint", "S0"]

# desired save order
SAVE_ORDER = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]

# net constraint order used in deep.Net.forward
LIB_CONS_ORDER = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]

# FINAL agreed ranges
PARAM_RANGES = {
    "Dpar": (0.0001, 0.0015),
    "Fint": (0.005, 0.4),
    "Dint": (0.0015, 0.004),
    "Fmv":  (0.0,   0.05),
    "Dmv":  (0.004, 0.2),
    "S0":   (0.90,  1.10),
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _align_bvals(bvals, nvols):
    b = np.asarray(bvals, dtype=float).ravel()
    if b.size == nvols:
        return b
    if nvols % b.size == 0:
        return np.tile(b, nvols // b.size)
    raise ValueError(f"bvals count ({b.size}) does not match volumes ({nvols}).")


def _qc_voxels(arr):
    finite = np.isfinite(arr).all(axis=1)
    positive_half = (arr > 0).sum(axis=1) >= (arr.shape[1] // 2)
    return finite & positive_half


def _reorder_params(A, out_order, save_order):
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"Expected 2D parameter array, got {A.shape}")

    if A.shape[1] == len(out_order):
        idx = [out_order.index(n) for n in save_order]
        return A[:, idx]
    if A.shape[0] == len(out_order):
        idx = [out_order.index(n) for n in save_order]
        return A[idx, :].T

    raise ValueError(f"Unexpected shape {A.shape}")


def _clamp_params(A, save_order):
    A = A.copy()
    for j, name in enumerate(save_order):
        lo, hi = PARAM_RANGES[name]
        A[:, j] = np.clip(A[:, j], lo, hi)
    return A


def _apply_runtime_config(arg, ir, te, tr, ti):
    # IR toggle everywhere relevant
    arg.net_pars.IR = bool(ir)
    arg.net_pars.fitS0 = True
    arg.sim.IR = bool(ir)

    # Paper (Voorter et al. 2023): sigmoid on D-params, abs on fractions
    arg.net_pars.con = "sigmoidabs"

    # enforce final agreed bounds in correct order
    arg.net_pars.cons_min = [PARAM_RANGES[n][0] for n in LIB_CONS_ORDER]
    arg.net_pars.cons_max = [PARAM_RANGES[n][1] for n in LIB_CONS_ORDER]

    # timing
    arg.rel_times.echotime = float(te)
    arg.rel_times.repetitiontime = float(tr)
    arg.rel_times.inversiontime = float(ti)


def _save_maps(A, mask3d, ref_img, out_dir, names, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = mask3d.astype(bool)
    nvox = int(mask.sum())
    p = len(names)

    if A.shape != (nvox, p):
        raise ValueError(f"{tag}: got {A.shape}, expected {(nvox, p)}")

    # individual maps
    for j, name in enumerate(names):
        vol = np.zeros(mask.shape, dtype=np.float32)
        vol[mask] = A[:, j].astype(np.float32)
        nib.save(nib.Nifti1Image(vol, ref_img.affine, ref_img.header), str(out_dir / f"{tag}_{name}.nii.gz"))

    # 4D stack
    stack = np.zeros(mask.shape + (p,), dtype=np.float32)
    for j in range(p):
        vol = np.zeros(mask.shape, dtype=np.float32)
        vol[mask] = A[:, j].astype(np.float32)
        stack[..., j] = vol
    nib.save(nib.Nifti1Image(stack, ref_img.affine, ref_img.header), str(out_dir / f"{tag}_params_4d.nii.gz"))

    with open(out_dir / f"{tag}_ORDER.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(names))


def main():
    parser = argparse.ArgumentParser(description="Apply trained PINN model on 4D DWI (7T).")
    parser.add_argument("--input", required=True, type=Path, help="4D DWI NIfTI")
    parser.add_argument("--mask", required=True, type=Path, help="3D mask NIfTI")
    parser.add_argument("--bvals", required=True, type=Path, help="bvals txt")
    parser.add_argument("--model", required=True, type=Path, help="trained model .pt")
    parser.add_argument("--out", required=True, type=Path, help="output directory")

    parser.add_argument("--ir", required=True, type=str2bool, help="true/false")
    parser.add_argument("--te", required=True, type=float, help="TE (ms)")
    parser.add_argument("--tr", required=True, type=float, help="TR (ms)")
    parser.add_argument("--ti", required=True, type=float, help="TI (ms), used when IR=true")

    parser.add_argument("--run-nnls", type=str2bool, default=True, help="also run NNLS reference")
    args = parser.parse_args()

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load data
    img = nib.load(str(args.input))
    data = img.get_fdata()
    mask = nib.load(str(args.mask)).get_fdata() > 0

    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.shape}")
    if mask.shape != data.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} != data shape {data.shape[:3]}")

    flat = data[mask]  # [N, V]
    bvals = _align_bvals(np.loadtxt(str(args.bvals)).ravel(), data.shape[3])

    keep = _qc_voxels(flat)
    trace = flat[keep]
    if trace.shape[0] == 0:
        raise ValueError("No voxels left after QC.")

    # Setup args
    arg = hp_example()
    arg = deep.checkarg(arg)
    _apply_runtime_config(arg, args.ir, args.te, args.tr, args.ti)

    # Load model
    device = arg.train_pars.device
    b_tensor = torch.as_tensor(bvals.astype(np.float32), dtype=torch.float32, device=device)

    net = deep.Net(b_tensor, arg.net_pars, arg.rel_times).to(device)
    state = torch.load(args.model, map_location=device)
    net.load_state_dict(state)
    net.eval()

    # PINN inference
    pred_pinn = deep.predict_IVIM(trace, bvals, net, arg)
    P_pinn = _reorder_params(np.asarray(pred_pinn), LIB_OUT_ORDER, SAVE_ORDER)
    P_pinn = _clamp_params(P_pinn, SAVE_ORDER)

    # Inflate + save PINN
    n_all = int(mask.sum())
    p = len(SAVE_ORDER)
    P_pinn_full = np.full((n_all, p), np.nan, dtype=np.float32)
    P_pinn_full[keep] = P_pinn.astype(np.float32)
    _save_maps(P_pinn_full, mask, img, args.out, SAVE_ORDER, "PINN")

    # Optional NNLS reference
    if args.run_nnls:
        pred_nnls = fit.fit_dats(
            bvals,
            trace,
            getattr(arg, "fit", None),
            method="NNLS",
            IR=bool(args.ir),
            rel_times=arg.rel_times,
        )
        P_nnls = _reorder_params(np.asarray(pred_nnls), LIB_OUT_ORDER, SAVE_ORDER)
        P_nnls = _clamp_params(P_nnls, SAVE_ORDER)

        P_nnls_full = np.full((n_all, p), np.nan, dtype=np.float32)
        P_nnls_full[keep] = P_nnls.astype(np.float32)
        _save_maps(P_nnls_full, mask, img, args.out, SAVE_ORDER, "NNLS")

    # Metadata
    meta = {
        "field_strength": "7T",
        "ir": bool(args.ir),
        "te_ms": float(args.te),
        "tr_ms": float(args.tr),
        "ti_ms": float(args.ti),
        "constraint": arg.net_pars.con,
        "ranges": PARAM_RANGES,
        "save_order": SAVE_ORDER,
        "model": str(args.model),
        "input": str(args.input),
        "mask": str(args.mask),
        "bvals": str(args.bvals),
    }
    with open(args.out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Done in {(time.time() - t0)/60:.2f} min")


if __name__ == "__main__":
    main()