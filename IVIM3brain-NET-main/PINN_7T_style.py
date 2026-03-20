#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7T-optimized IVIM / DTI-IVIM — Apply pretrained PINN + NNLS (+ optional LSQ)
Paulien-style: training is done elsewhere; this runner only loads and applies the model.
"""

import os, time
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

# IVIMNET
import IVIMNET.deep as deep
import IVIMNET.fitting_algorithms as fit
from hyperparams import hyperparams as hp_example

# ---------- CONFIG ----------
ROOT       = Path(r"C:\Users\mtahi\IVIM-DTI-NET-main\data\subject05_ks88\Data\ti-off")
INPUT_4D   = ROOT / "TORTOISE_ti-off.nii.gz"
MASK_3D    = ROOT / "mask.nii.gz"
BVALS_TXT  = ROOT / "AP.bval"
OUT_DIR    = ROOT / "final_preprocessing"
MODEL_PT   = Path(r"C:\Users\mtahi\IVIM-DTI-NET-main\data\subject05_ks88\Data\ti-off\final_preprocessing\results_training_7T_ti-off\PINN_7T_trained_ti-off.pt")
DO_LSQ = False  # optional reference

# Library output order (must match IVIMNET.deep.predict_IVIM)
LIB_OUT_ORDER = ["Dpar", "Fmv", "Dmv", "Dint", "Fint", "S0"]
# How you save maps on disk
SAVE_ORDER    = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]
# Constraint order must match deep.forward() indexing
LIB_CONS_ORDER = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]

# 7T physiology-aware parameter ranges
PARAM_RANGES_7T = {
    "Dpar": (0.0001, 0.0015),
    "Fint": (0.0,    0.40), # set it to 0.4 for IR data
    "Dint": (0.0015, 0.004),
    "Fmv":  (0.0,    0.05),
    "Dmv":  (0.004,  0.20),
    "S0":   (0.90,   1.10),
}

# Acquisition timings (ms) — adjust to your protocol
class RelaxationTimes7T:
    TE = 50
    TR = 4500
    TI = 0 #2200 for IR data
    T2_blood  = 23
    T2_tissue = 46
    T2_isf    = 100
    T1_blood  = 2600
    T1_tissue = 1200
    T1_isf    = 4300

rt7 = RelaxationTimes7T()

# ---------------------------- helpers ----------------------------
def _push_times(arg, te_ms, tr_ms, ti_ms):
    if not hasattr(arg, "rel_times"):
        class _Rel: pass
        arg.rel_times = _Rel()
    arg.rel_times.echotime       = te_ms
    arg.rel_times.repetitiontime = tr_ms
    arg.rel_times.inversiontime  = ti_ms

    if hasattr(arg, "sim"):
        arg.sim.echotime       = te_ms
        arg.sim.repetitiontime = tr_ms
        arg.sim.inversiontime  = ti_ms
        arg.sim.IR = False # True for IR (inversion recovery); False for standard

    if hasattr(arg, "net_pars"):
        arg.net_pars.IR = False # True for IR (inversion recovery); False for standard
        arg.net_pars.fitS0 = True

def _apply_7t_relaxations(arg, rt):
    # write into arg.sim (if present) and arg.rel_times (used by forward model)
    mapping = dict(
        bloodT2=rt.T2_blood, tissueT2=rt.T2_tissue, isfT2=rt.T2_isf,
        bloodT1=rt.T1_blood, tissueT1=rt.T1_tissue, isfT1=rt.T1_isf,
    )
    if hasattr(arg, "sim"):
        for k, v in mapping.items():
            setattr(arg.sim, k, v)
    if hasattr(arg, "rel_times"):
        for k, v in mapping.items():
            setattr(arg.rel_times, k, v)

def _apply_constraints_7t(arg):
    mins = np.array([PARAM_RANGES_7T[n][0] for n in LIB_CONS_ORDER], dtype=float)
    maxs = np.array([PARAM_RANGES_7T[n][1] for n in LIB_CONS_ORDER], dtype=float)
    arg.net_pars.cons_min = mins.tolist()
    arg.net_pars.cons_max = maxs.tolist()
    print("✅ Using constraints (order = {}):".format(LIB_CONS_ORDER))
    print("   mins:", arg.net_pars.cons_min)
    print("   maxs:", arg.net_pars.cons_max)

def _align_bvals(bvals, nvols):
    b = np.asarray(bvals, dtype=float).ravel()
    if b.size == nvols:
        return b
    if nvols % b.size == 0:
        return np.tile(b, nvols // b.size)
    raise ValueError(f"bvals count ({b.size}) doesn't match volumes ({nvols}).")

def _normalize_by_b0(arr, bvals):
    b = np.asarray(bvals, dtype=float).ravel()
    if not np.any(b == 0):
        raise ValueError("No b=0 volumes in bvals; required for S0 normalization.")
    s0 = arr[:, b == 0].mean(axis=1, keepdims=True)
    s0[s0 == 0] = 1.0
    return arr / s0

def _qc_voxels(arr):
    finite = np.isfinite(arr).all(axis=1)
    positive_half = (arr > 0).sum(axis=1) >= (arr.shape[1] // 2)
    return finite & positive_half

def _reorder_params(A, out_order, save_order):
    A = np.asarray(A)
    P = len(out_order)
    if A.shape[1] == P:
        idx = [out_order.index(name) for name in save_order]
        return A[:, idx]
    elif A.shape[0] == P:
        idx = [out_order.index(name) for name in save_order]
        return A[idx, :].T
    else:
        raise ValueError(f"Unexpected param shape {A.shape}; cannot reorder.")

def _clamp_fracs(A, save_order):
    A = A.copy()
    for pname, (lo, hi) in dict(Fint=(0.0,0.7), Fmv=(0.0,0.05)).items(): # set fint 0.4 for IR data
        j = save_order.index(pname)
        A[:, j] = np.clip(A[:, j], lo, hi)
    return A

def validate_params(params, method_name):
    for i, name in enumerate(SAVE_ORDER):
        min_val, max_val = PARAM_RANGES_7T[name]
        within_range = np.sum((params[:, i] >= min_val) & (params[:, i] <= max_val))
        pct = within_range / len(params) * 100
        print(f"{method_name:<6} {name:<5}: {pct:5.1f}% within expected range")


def _save_maps(A, mask3d, ref_img, out_dir, names, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = mask3d.astype(bool)
    N, P = mask.sum(), len(names)
    assert A.shape == (N, P), f"Bad shape {A.shape} vs {(N,P)}"
    # per-parameter NIfTI
    for j, name in enumerate(names):
        vol = np.zeros(mask.shape, dtype=np.float32)
        vol[mask] = A[:, j].astype(np.float32)
        nib.save(nib.Nifti1Image(vol, ref_img.affine, ref_img.header),
                 str(out_dir / f"{tag}_{name}.nii.gz"))
    # 4D stack
    stack = np.zeros(mask.shape + (P,), dtype=np.float32)
    for j in range(P):
        vol = np.zeros(mask.shape, dtype=np.float32)
        vol[mask] = A[:, j].astype(np.float32)
        stack[..., j] = vol
    nib.save(nib.Nifti1Image(stack, ref_img.affine, ref_img.header),
             str(out_dir / f"{tag}_params_4d.nii.gz"))
    with open(out_dir / f"{tag}_ORDER.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(names))

# ---------------------------- runner ----------------------------
def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    img  = nib.load(str(INPUT_4D))
    data = img.get_fdata()
    mask = nib.load(str(MASK_3D)).get_fdata() > 0
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.shape}.")
    if mask.shape != data.shape[:3]:
        raise ValueError(f"Mask shape {mask.shape} != data XYZ {data.shape[:3]}.")

    nvols = data.shape[3]
    flat  = data[mask]   # (N, V)
    print(f"Masked voxels: {flat.shape[0]} | Volumes: {nvols}")

    # --- bvals ---
    bvals_file = np.loadtxt(str(BVALS_TXT)).ravel()
    bvals = _align_bvals(bvals_file, nvols)

    # --- Normalize & QC ---
    flat  = flat  #_normalize_by_b0(flat, bvals)
    keep  = _qc_voxels(flat)
    trace = flat[keep]
    print(f"After QC: {trace.shape[0]} voxels retained.")

    # --- Hyperparams & 7T physiology ---
    arg = hp_example()
    arg = deep.checkarg(arg)
    if hasattr(arg, "sim"):  # real data mode
        arg.sim.simulate = True
    _push_times(arg, te_ms=rt7.TE, tr_ms=rt7.TR, ti_ms=rt7.TI)
    _apply_7t_relaxations(arg, rt7)
    _apply_constraints_7t(arg)

    # --- Load pretrained model ---
    device = arg.train_pars.device
    b = bvals.astype(np.float32) 
    b_tensor = torch.as_tensor(b, dtype=torch.float32, device=device)
    net = deep.Net(b_tensor, arg.net_pars, arg.rel_times).to(device)

    state = torch.load(MODEL_PT, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print(f"✅ Loaded pretrained PINN: {MODEL_PT} on {device}")


    # --- PINN prediction ---
    pred_pinn = deep.predict_IVIM(trace, b, net, arg)
    P_pinn = _reorder_params(np.asarray(pred_pinn), LIB_OUT_ORDER, SAVE_ORDER)
    P_pinn = _clamp_fracs(P_pinn, SAVE_ORDER)

    # --- NNLS (reference) ---
    print("Running NNLS...")
    pred_nnls = fit.fit_dats(b, trace, getattr(arg, "fit", None), "NNLS", IR=False) # True for IR (inversion recovery); False for standard
    P_nnls = _reorder_params(np.asarray(pred_nnls), LIB_OUT_ORDER, SAVE_ORDER)
    P_nnls = _clamp_fracs(P_nnls, SAVE_ORDER)

    # --- Optional LSQ ---
    if DO_LSQ:
        pred_lsq = fit.fit_dats(b, trace, getattr(arg, "fit", None), "two-step-lsq", IR=True)
        P_lsq = _reorder_params(np.asarray(pred_lsq), LIB_OUT_ORDER, SAVE_ORDER)
        P_lsq = _clamp_fracs(P_lsq, SAVE_ORDER)
    else:
        P_lsq = None

    # --- Inflate back to whole mask and save ---
    N_all = mask.sum()
    def _inflate(Asmall):
        A = np.full((N_all, len(SAVE_ORDER)), np.nan, dtype=np.float32)
        A[keep] = Asmall.astype(np.float32)
        return A

    P_pinn_full = _inflate(P_pinn)
    P_nnls_full = _inflate(P_nnls)
    _save_maps(P_pinn_full, mask, img, OUT_DIR, SAVE_ORDER, "PINN")
    _save_maps(P_nnls_full, mask, img, OUT_DIR, SAVE_ORDER, "NNLS")

    if P_lsq is not None:
        P_lsq_full = _inflate(P_lsq)
        _save_maps(P_lsq_full, mask, img, OUT_DIR, SAVE_ORDER, "LSQ")

    print(f"✅ Done in {(time.time()-t0)/60:.1f} min. Saved: {', '.join(SAVE_ORDER)}")

if __name__ == "__main__":
    main()
