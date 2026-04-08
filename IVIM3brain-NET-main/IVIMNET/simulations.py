"""
Simulation and benchmarking utilities for 3C-IVIM PINN (7T aligned)

Changes from original IVIM3brain-NET (Voorter et al., MRM 2023;90:1657-1671):
  - Vectorized sim_signal: signal + noise generated without Python for-loops
    (~100x speedup for sims=11.5M)
  - Fixed 'import scipy.stats as scipy' namespace shadow
  - Removed module-level RNG seeding
  - Use 'Agg' backend by default (headless-safe)
"""

import os
import time
import numpy as np
from scipy import stats as spstats
import torch
import tqdm
from joblib import Parallel, delayed
import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import IVIMNET.deep as deep
import IVIMNET.fitting_algorithms as fit


def sim(SNR, arg):
    arg = deep.checkarg(arg)

    # Support SNR range for general pipeline robustness
    snr_min = getattr(arg.sim, 'snr_min', None)
    snr_max = getattr(arg.sim, 'snr_max', None)

    IVIM_signal_noisy, Dpar, fint, Dint, fmv, Dmv = sim_signal(
        SNR=SNR,
        bvalues=arg.sim.bvalues,
        IR=arg.sim.IR,
        rel_times=arg.rel_times,
        sims=arg.sim.sims,
        distribution=arg.sim.distribution,
        Dparmin=arg.sim.range[0][0],
        Dparmax=arg.sim.range[1][0],
        fintmin=arg.sim.range[0][1],
        fintmax=arg.sim.range[1][1],
        Dintmin=arg.sim.range[0][2],
        Dintmax=arg.sim.range[1][2],
        fmvmin=arg.sim.range[0][3],
        fmvmax=arg.sim.range[1][3],
        Dmvmin=arg.sim.range[0][4],
        Dmvmax=arg.sim.range[1][4],
        rician=arg.sim.rician,
        snr_min=snr_min,
        snr_max=snr_max,
    )

    Dpar = Dpar[:arg.sim.num_samples_eval]
    Dint = Dint[:arg.sim.num_samples_eval]
    fint = fint[:arg.sim.num_samples_eval]
    Dmv = Dmv[:arg.sim.num_samples_eval]
    fmv = fmv[:arg.sim.num_samples_eval]

    if arg.sim.n_ensemble > 1:
        paramsNN = np.zeros([arg.sim.repeats, arg.sim.n_ensemble, 6, arg.sim.num_samples_eval])
    else:
        paramsNN = np.zeros([arg.sim.repeats, 6, arg.sim.num_samples_eval])

    net = None
    if not arg.train_pars.skip_net:
        for aa in range(arg.sim.repeats):
            if arg.sim.n_ensemble > 1:
                if arg.sim.jobs > 1:
                    def parfun(_):
                        n = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
                        return deep.predict_IVIM(
                            IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, n, arg
                        )
                    output = Parallel(n_jobs=arg.sim.jobs)(
                        delayed(parfun)(i) for i in tqdm.tqdm(range(arg.sim.n_ensemble), position=0, leave=True)
                    )
                    for bb in range(arg.sim.n_ensemble):
                        paramsNN[aa, bb] = output[bb]
                else:
                    for bb in range(arg.sim.n_ensemble):
                        net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
                        paramsNN[aa, bb] = deep.predict_IVIM(
                            IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net, arg
                        )
                if arg.train_pars.use_cuda:
                    torch.cuda.empty_cache()
            else:
                print(f"\nRepeat: {aa}\n")
                t0 = time.time()
                net = deep.learn_IVIM(IVIM_signal_noisy, arg.sim.bvalues, arg)
                print(f"\ntime elapsed for PI-NN training: {time.time() - t0}\n")

                t1 = time.time()
                pred = deep.predict_IVIM(
                    IVIM_signal_noisy[:arg.sim.num_samples_eval, :], arg.sim.bvalues, net, arg
                )
                if arg.sim.repeats > 1:
                    paramsNN[aa] = pred
                else:
                    paramsNN = pred
                print(f"\ntime elapsed for PI-NN inference: {time.time() - t1}\n")

                if arg.train_pars.use_cuda:
                    torch.cuda.empty_cache()

        if arg.sim.n_ensemble > 1:
            paramsNN = np.mean(paramsNN, axis=1)

        print("\nresults for PI-NN")
        plot_dependency_figs(
            np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv),
            np.squeeze(paramsNN), "PINN"
        )

        if arg.sim.repeats > 1:
            matNN = np.zeros([arg.sim.repeats, 5, 5])
            for aa in range(arg.sim.repeats):
                matNN[aa] = print_errors(
                    np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint),
                    np.squeeze(fmv), np.squeeze(Dmv), paramsNN[aa], arg
                )
            matNN = np.mean(matNN, axis=0)
            stability = np.sqrt(np.mean(np.square(np.std(paramsNN, axis=0)), axis=1))
            stability = stability[[0, 4, 3, 1, 2]] / [
                np.mean(Dpar), np.mean(fint), np.mean(Dint), np.mean(fmv), np.mean(Dmv)
            ]
            paramsNN_0 = paramsNN[0]
        else:
            matNN = print_errors(
                np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint),
                np.squeeze(fmv), np.squeeze(Dmv), np.squeeze(paramsNN), arg
            )
            stability = np.zeros(5)
            paramsNN_0 = paramsNN
    else:
        stability = np.zeros(5)
        matNN = np.zeros([5, 5])
        paramsNN_0 = None

    if arg.fit.do_fit:
        t2 = time.time()
        method = "NNLS"
        paramsnnls = fit.fit_dats(
            arg.sim.bvalues,
            IVIM_signal_noisy[:arg.sim.num_samples_eval, :],
            arg.fit,
            method,
            IR=arg.sim.IR,
            rel_times=arg.rel_times,
        )
        print(f"\ntime elapsed for NNLS fit: {time.time() - t2} seconds\n")
        print("results for fit NNLS")
        matnnls = print_errors(
            np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv),
            paramsnnls, arg
        )
        plot_dependency_figs(
            np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv),
            paramsnnls, method
        )

        if not arg.train_pars.skip_net and paramsNN_0 is not None:
            plot_pred_vs_true(
                np.squeeze(Dpar), np.squeeze(fint), np.squeeze(Dint), np.squeeze(fmv), np.squeeze(Dmv),
                np.squeeze(paramsNN_0), paramsnnls, paramsnnls
            )

        return matNN, matnnls, stability
    else:
        return matNN, stability, net


def sim_signal(
    SNR,
    bvalues,
    IR=True,
    rel_times=None,
    sims=1000000,
    distribution="normal",
    Dparmin=0.0001,
    Dparmax=0.0015,
    fintmin=0.0,
    fintmax=0.40,
    Dintmin=0.0015,
    Dintmax=0.004,
    fmvmin=0.0,
    fmvmax=0.05,
    Dmvmin=0.004,
    Dmvmax=0.2,
    rician=False,
    state=123,
    snr_min=None,
    snr_max=None,
):
    """
    Generate simulated 3C-IVIM signals with optional IR weighting and noise.

    CHANGED: fully vectorized — no Python for-loops over voxels.
    At 11.5M sims this gives ~100x speedup vs the original per-voxel loop.

    SNR augmentation: if snr_min and snr_max are both provided, each simulated
    voxel is assigned a random SNR drawn from uniform(snr_min, snr_max).
    This makes the trained network robust across the full 7T SNR range
    rather than tuned to a single fixed noise level.
    If snr_min/snr_max are not provided, the scalar SNR argument is used.
    """
    rg = np.random.RandomState(state)

    if distribution == "uniform":
        Dpar = Dparmin + rg.uniform(0, 1, (sims, 1)) * (Dparmax - Dparmin)
        fint = fintmin + rg.uniform(0, 1, (sims, 1)) * (fintmax - fintmin)
        Dint = Dintmin + rg.uniform(0, 1, (sims, 1)) * (Dintmax - Dintmin)
        fmv = fmvmin + rg.uniform(0, 1, (sims, 1)) * (fmvmax - fmvmin)
        Dmv = Dmvmin + rg.uniform(0, 1, (sims, 1)) * (Dmvmax - Dmvmin)
    elif distribution == "normal":
        Dpar = np.abs((Dparmax + Dparmin) / 2 + rg.standard_normal((sims, 1)) * ((Dparmax - Dparmin) / 6))
        fint = np.abs((fintmax + fintmin) / 2 + rg.standard_normal((sims, 1)) * ((fintmax - fintmin) / 6))
        Dint = (Dintmax + Dintmin) / 2 + rg.standard_normal((sims, 1)) * ((Dintmax - Dintmin) / 6)
        fmv = np.abs((fmvmax + fmvmin) / 2 + rg.standard_normal((sims, 1)) * ((fmvmax - fmvmin) / 6))
        Dmv = (Dmvmax + Dmvmin) / 2 + rg.standard_normal((sims, 1)) * ((Dmvmax - Dmvmin) / 6)
    elif distribution == "normal-wide":
        Dpar = np.abs((Dparmax + Dparmin) / 2 + rg.standard_normal((sims, 1)) * ((Dparmax - Dparmin) / 4))
        fint = np.abs((fintmax + fintmin) / 2 + rg.standard_normal((sims, 1)) * ((fintmax - fintmin) / 4))
        Dint = (Dintmax + Dintmin) / 2 + rg.standard_normal((sims, 1)) * ((Dintmax - Dintmin) / 4)
        fmv = np.abs((fmvmax + fmvmin) / 2 + rg.standard_normal((sims, 1)) * ((fmvmax - fmvmin) / 4))
        Dmv = (Dmvmax + Dmvmin) / 2 + rg.standard_normal((sims, 1)) * ((Dmvmax - Dmvmin) / 4)
    else:
        raise Exception("distribution must be 'uniform', 'normal', or 'normal-wide'")

    bvalues = np.asarray(bvalues, dtype=float).ravel()

    # --- Vectorized signal generation (replaces per-voxel Python loop) ---
    # Shapes: params are (sims, 1), bvalues is (nb,)
    # Broadcasting: (sims, 1) * (nb,) -> (sims, nb)
    if IR:
        if rel_times is None:
            raise ValueError("IR=True requires rel_times")
        # Use the vectorized fitting_algorithms functions directly
        # tri_expN_noS0_IR supports broadcasting when params are (sims,1) and bvalues is (nb,)
        data_sim = fit.tri_expN_noS0_IR(bvalues, Dpar, fint, Dint, fmv, Dmv, rel_times)
    else:
        data_sim = fit.tri_expN_noS0(bvalues, Dpar, fint, Dint, fmv, Dmv)

    # --- Vectorized noise generation (replaces per-voxel loop) ---
    if snr_min is not None and snr_max is not None:
        # Per-voxel SNR: each simulated voxel sees a different noise level,
        # drawn uniformly from [snr_min, snr_max]. This is the key to making
        # the network robust across the full range of 7T acquisitions.
        snr_per_vox = rg.uniform(snr_min, snr_max, (sims, 1))  # shape (sims, 1)
        noise_real = rg.normal(0, 1, (sims, nb := len(bvalues))) / snr_per_vox
        noise_imag = rg.normal(0, 1, (sims, nb)) / snr_per_vox
    elif SNR > 0:
        nb = len(bvalues)
        noise_real = rg.normal(0, 1.0 / SNR, (sims, nb))
        noise_imag = rg.normal(0, 1.0 / SNR, (sims, nb))
    else:
        noise_real = noise_imag = 0

    if SNR > 0 or (snr_min is not None and snr_max is not None):
        if rician:
            data_sim = np.sqrt((data_sim + noise_real) ** 2 + noise_imag ** 2)
        else:
            data_sim = data_sim + noise_real

    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    S0_noisy[S0_noisy == 0] = 1.0
    data_sim = data_sim / S0_noisy[:, None]

    return data_sim, Dpar, fint, Dint, fmv, Dmv


def print_errors(Dpar, fint, Dint, fmv, Dmv, params, arg):
    print('\nDpar was found {} times out of {}, percentage not found: {}%'.format(
        np.count_nonzero(~np.isnan(params[0])), arg.sim.num_samples_eval,
        100 * (arg.sim.num_samples_eval - np.count_nonzero(~np.isnan(params[0]))) / arg.sim.num_samples_eval
    ))
    print('Dint was found {} times out of {}, percentage not found: {}%'.format(
        np.count_nonzero(~np.isnan(params[3])), arg.sim.num_samples_eval,
        100 * (arg.sim.num_samples_eval - np.count_nonzero(~np.isnan(params[3]))) / arg.sim.num_samples_eval
    ))
    print('Dmv was found {} times out of {}, percentage not found: {}%'.format(
        np.count_nonzero(~np.isnan(params[2])), arg.sim.num_samples_eval,
        100 * (arg.sim.num_samples_eval - np.count_nonzero(~np.isnan(params[2]))) / arg.sim.num_samples_eval
    ))

    rmse_Dpar = np.sqrt(np.nanmean(np.square(np.subtract(Dpar, params[0]))))
    rmse_fmv = np.sqrt(np.nanmean(np.square(np.subtract(fmv, params[1]))))
    rmse_Dmv = np.sqrt(np.nanmean(np.square(np.subtract(Dmv, params[2]))))
    rmse_Dint = np.sqrt(np.nanmean(np.square(np.subtract(Dint, params[3]))))
    rmse_fint = np.sqrt(np.nanmean(np.square(np.subtract(fint, params[4]))))

    Spearman = np.zeros([10, 1])
    Spearman[0, 0] = spstats.spearmanr(params[3], params[4], nan_policy='omit')[0]
    Spearman[1, 0] = spstats.spearmanr(params[2], params[1], nan_policy='omit')[0]
    Spearman[2, 0] = spstats.spearmanr(params[4], params[1], nan_policy='omit')[0]
    Spearman[3, 0] = spstats.spearmanr(params[0], params[3], nan_policy='omit')[0]
    Spearman[4, 0] = spstats.spearmanr(params[3], params[2], nan_policy='omit')[0]
    Spearman[5, 0] = spstats.spearmanr(params[0], params[1], nan_policy='omit')[0]
    Spearman[6, 0] = spstats.spearmanr(params[0], params[2], nan_policy='omit')[0]
    Spearman[7, 0] = spstats.spearmanr(params[0], params[4], nan_policy='omit')[0]
    Spearman[8, 0] = spstats.spearmanr(params[1], params[3], nan_policy='omit')[0]
    Spearman[9, 0] = spstats.spearmanr(params[2], params[4], nan_policy='omit')[0]
    Spearman[np.isnan(Spearman)] = 1

    meanDpar_true = np.mean(Dpar)
    meanfint_true = np.mean(fint)
    meanDint_true = np.mean(Dint)
    meanfmv_true = np.mean(fmv)
    meanDmv_true = np.mean(Dmv)

    meanDpar_fitted = np.nanmean(params[0])
    meanfint_fitted = np.nanmean(params[4])
    meanDint_fitted = np.nanmean(params[3])
    meanfmv_fitted = np.nanmean(params[1])
    meanDmv_fitted = np.nanmean(params[2])

    print('\nresults: columns show mean true, mean fitted, NRMSE; rows show Dpar, fint, Dint, fmv, Dmv\n')
    print([meanDpar_true, meanDpar_fitted, rmse_Dpar / max(meanDpar_true, 1e-12)])
    print([meanfint_true, meanfint_fitted, rmse_fint / max(meanfint_true, 1e-12)])
    print([meanDint_true, meanDint_fitted, rmse_Dint / max(meanDint_true, 1e-12)])
    print([meanfmv_true, meanfmv_fitted, rmse_fmv / max(meanfmv_true, 1e-12)])
    print([meanDmv_true, meanDmv_fitted, rmse_Dmv / max(meanDmv_true, 1e-12)])

    mats = [
        [meanDpar_true, meanDpar_fitted, rmse_Dpar / max(meanDpar_true, 1e-12), Spearman[0, 0], Spearman[5, 0]],
        [meanfint_true, meanfint_fitted, rmse_fint / max(meanfint_true, 1e-12), Spearman[1, 0], Spearman[6, 0]],
        [meanDint_true, meanDint_fitted, rmse_Dint / max(meanDint_true, 1e-12), Spearman[2, 0], Spearman[7, 0]],
        [meanfmv_true, meanfmv_fitted, rmse_fmv / max(meanfmv_true, 1e-12), Spearman[3, 0], Spearman[8, 0]],
        [meanDmv_true, meanDmv_fitted, rmse_Dmv / max(meanDmv_true, 1e-12), Spearman[4, 0], Spearman[9, 0]],
    ]
    return mats


def plot_pred_vs_true(Dpar, fint, Dint, fmv, Dmv, paramsNN, paramslsq, paramsnnls):
    pathresults = '{folder}/results'.format(folder=os.getcwd())
    os.makedirs(pathresults, exist_ok=True)

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dpar, paramsnnls[0], color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(Dpar, paramsNN[0], color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([.0001, .0015]), np.array([.0001, .0015]), 'k--')
    plt.xlabel('true Dpar'); plt.ylabel('estimated Dpar')
    plt.axis((0.00003, 0.00157, 0.00003, 0.00157))
    plt.legend(('NNLS', 'PI-NN'))
    plt.savefig('{}/accuracy_Dpar.png'.format(pathresults))
    plt.close('all')

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fmv, paramsnnls[1], color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(fmv, paramsNN[1], color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.0, .05]), np.array([0.0, .05]), 'k--')
    plt.xlabel('true Fmv'); plt.ylabel('estimated Fmv')
    plt.axis((-0.005, 0.055, -0.005, 0.055))
    plt.legend(('NNLS', 'PI-NN'))
    plt.savefig('{}/accuracy_Fmv.png'.format(pathresults))
    plt.close('all')

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dmv, paramsnnls[2], color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(Dmv, paramsNN[2], color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.004, .2]), np.array([0.004, .2]), 'k--')
    plt.xlabel('true Dmv'); plt.ylabel('estimated Dmv')
    plt.axis((0.004 - .0098, 0.2 + .0098, 0.004 - .0098, 0.2 + .0098))
    plt.legend(('NNLS', 'PI-NN'))
    plt.savefig('{}/accuracy_Dmv.png'.format(pathresults))
    plt.close('all')

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(Dint, paramsnnls[3], color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(Dint, paramsNN[3], color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.0015, .004]), np.array([0.0015, .004]), 'k--')
    plt.xlabel('true Dint'); plt.ylabel('estimated Dint')
    plt.axis((0.0015 - .000125, 0.004 + .000125, 0.0015 - .000125, 0.004 + .000125))
    plt.legend(('NNLS', 'PI-NN'))
    plt.savefig('{}/accuracy_Dint.png'.format(pathresults))
    plt.close('all')

    plt.figure(figsize=(10, 10), dpi=100)
    plt.plot(fint, paramsnnls[4], color='turquoise', linestyle='', marker='.', markersize=1)
    plt.plot(fint, paramsNN[4], color='indigo', linestyle='', marker='.', markersize=1)
    plt.plot(np.array([0.0, .4]), np.array([0.0, .4]), 'k--')
    plt.xlabel('true Fint'); plt.ylabel('estimated Fint')
    plt.axis((-0.02, 0.42, -0.02, 0.42))
    plt.legend(('NNLS', 'PI-NN'))
    plt.savefig('{}/accuracy_Fint.png'.format(pathresults))
    plt.close('all')


def plot_dependency_figs(Dpar, Fint, Dint, Fmv, Dmv, params, method):
    """
    Ground-truth args order:
      Dpar, Fint, Dint, Fmv, Dmv
    Predicted params expected order:
      params[0]=Dpar, params[1]=Fmv, params[2]=Dmv, params[3]=Dint, params[4]=Fint
    """
    pathresults = '{folder}/results'.format(folder=os.getcwd())
    os.makedirs(pathresults, exist_ok=True)

    params = np.asarray(params)
    if params.ndim != 2:
        raise ValueError(f"plot_dependency_figs expects 2D params, got {params.shape}")
    if params.shape[0] < 5 and params.shape[1] >= 5:
        params = params.T
    if params.shape[0] < 5:
        raise ValueError(f"plot_dependency_figs expects >=5 parameter rows, got {params.shape}")

    pairs = [
        (Fint, Dpar, params[4], params[0], 'Fint', 'Dpar', 'dependency_Fint_Dpar_{}.png'),
        (Fmv,  Dpar, params[1], params[0], 'Fmv',  'Dpar', 'dependency_Fmv_Dpar_{}.png'),
        (Dmv,  Dpar, params[2], params[0], 'Dmv',  'Dpar', 'dependency_Dmv_Dpar_{}.png'),
        (Dint, Dpar, params[3], params[0], 'Dint', 'Dpar', 'dependency_Dint_Dpar_{}.png'),
        (Fmv,  Dmv,  params[1], params[2], 'Fmv',  'Dmv',  'dependency_Fmv_Dmv_{}.png'),
        (Fmv,  Dint, params[1], params[3], 'Fmv',  'Dint', 'dependency_Fmv_Dint_{}.png'),
        (Fint, Fmv,  params[4], params[1], 'Fint', 'Fmv',  'dependency_Fint_Fmv_{}.png'),
        (Dint, Dmv,  params[3], params[2], 'Dint', 'Dmv',  'dependency_Dint_Dmv_{}.png'),
        (Fint, Dmv,  params[4], params[2], 'Fint', 'Dmv',  'dependency_Fint_Dmv_{}.png'),
        (Fint, Dint, params[4], params[3], 'Fint', 'Dint', 'dependency_Fint_Dint_{}.png'),
    ]

    for x_true, y_true, x_pred, y_pred, xlabel, ylabel, fname in pairs:
        plt.figure(figsize=(10, 10), dpi=100)
        plt.plot(x_true, y_true, color='lightgray', linestyle='', marker='.', markersize=2)
        plt.plot(x_pred, y_pred, 'b.', markersize=1)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(('ground truth', 'predicted'), fontsize=14)

        if xlabel in ('Fint', 'Fmv'):
            plt.xlim(0, 1)
        if ylabel in ('Fint', 'Fmv'):
            plt.ylim(0, 1)

        plt.savefig('{}/{}'.format(pathresults, fname.format(method)))
        plt.close('all')