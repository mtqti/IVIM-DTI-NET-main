#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7T-optimized PINN training — based on Paulien Voorter (2022)
Adapted for [Dpar, Fint, Dint, Fmv, Dmv, S0] output order
"""

import numpy as np
import IVIMNET.simulations as sim
import IVIMNET.deep as deep
from hyperparams import hyperparams as hp_example
import time, os
from matplotlib import pyplot as plt
from pathlib import Path

# ----------------------------
# Load hyperparameters
arg = hp_example()
arg = deep.checkarg(arg)
arg.train_pars.skip_net = False

# Use strict 7T constraints (your runner + hyperparams are aligned to this order!)
arg.net_pars.cons_min = [0.0001, 0.0,    0.0015, 0.0,   0.004, 0.9]
arg.net_pars.cons_max = [0.0015, 0.4,    0.004,  0.05,  0.2,   1.1]

# Training tweaks
arg.train_pars.lr    = 3e-5
arg.train_pars.maxit = 500

# Simulation params
arg.sim.SNR      = 60     # realistic for 7T
arg.sim.repeats  = 1      # number of simulation repeats
arg.sim.n_ensemble = 1    # ensemble training runs
arg.fit.do_fit   = False  # PINN only (skip LSQ/NNLS during training)

# ----------------------------
# Output folder
ROOT = Path(r"C:\Users\mtahi\IVIM-DTI-NET-main\data\subject05_ks88\Data\ti-off\final_preprocessing")
pathresults = ROOT / "results_training_7T_ti-off"
os.makedirs(pathresults, exist_ok=True)

# ----------------------------
# Run simulation + training
start_time = time.time()

print(f"\nTraining PINN with SNR={arg.sim.SNR}, lr={arg.train_pars.lr}, iters={arg.train_pars.maxit}")

matNN, stability, net = sim.sim(arg.sim.SNR, arg)

elapsed_time = (time.time() - start_time) / 60
print(f"\n✅ Training finished in {elapsed_time:.1f} minutes\n")

# Save trained PINN
   #import torch
   #b = torch.FloatTensor(arg.sim.bvalues)  # training uses sim.bvalues
   #net = deep.Net(b, arg.net_pars, arg.rel_times)
   #net.load_state_dict(torch.load("trained_model_temp.pt")) if os.patdata/subject04/IR-IVIM_New/results_PINN_final_7T_Pauline_sigmoid/Results.ziph.exists("trained_model_temp.pt") else None
   #torch.save(net.state_dict(), pathresults / "PINN_7T_trained_sigmoid.pt")
   #print(f"✅ Saved trained model to {pathresults}/PINN_7T_trained_sigmoid.pt")

   # Save trained PINN (use the trained net returned by sim.sim)
import torch
torch.save(net.state_dict(), pathresults / "PINN_7T_trained_ti-off.pt")
print(f"✅ Saved trained model to {pathresults}/PINN_7T_trained_ti-off.pt")

# Save simulation results
np.save(f"{pathresults}/results-PINN-lr{arg.train_pars.lr}_ensemble{arg.sim.n_ensemble}", matNN)

# ----------------------------
# Optional: correlation matrix plot
rhomatrix = [[1, round(matNN[3][3],2), round(matNN[1][4],2), round(matNN[2][4],2), round(matNN[0][4],2)],
             [round(matNN[3][3],2), 1, round(matNN[4][3],2), round(matNN[0][3],2), round(matNN[3][4],2)],
             [round(matNN[1][4],2), round(matNN[4][3],2), 1, round(matNN[4][4],2), round(matNN[1][3],2)],
             [round(matNN[2][4],2), round(matNN[0][3],2), round(matNN[4][4],2), 1, round(matNN[2][3],2)],
             [round(matNN[0][4],2), round(matNN[3][4],2), round(matNN[1][3],2), round(matNN[2][3],2), 1]]

fig, ax = plt.subplots()
params = ['Dpar','Fint','Dint','Fmv','Dmv']
im = ax.matshow(np.array(rhomatrix), cmap=plt.cm.BrBG, vmin=-1, vmax=1)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(params); ax.set_yticklabels(params)
for i in range(5):
    for j in range(5):
        c = rhomatrix[j][i]
        ax.text(i, j, str(c), va='center', ha='center')
plt.title("Pearson correlation matrix PINN")
plt.colorbar(im)
plt.savefig(f"{pathresults}/dependency_Pearson_corr_matrix_PINN.png")
plt.close()
