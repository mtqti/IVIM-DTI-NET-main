"""
Hyperparameters for 3C-IVIM PINNbrain)

Original: O. Gurney-Champion & M. Kaandorp (2020) | Adapted: P. Voorter (2022)
Amended: 2025 (7T-friendly bounds; fractions truly bounded; no widening)

Parameter order used by the network constraints:
    [Dpar, Fint, Dint, Fmv, Dmv, S0]
"""

import os
import torch
import numpy as np


# -------------------------- training params --------------------------
class train_pars:
    def __init__(self, nets):
        self.optim = 'adam'
        self.lr = 3e-5  #
        self.patience = 10
        self.batch_size = 128
        self.maxit = 1000
        self.split = 0.90
        self.load_nn = False  # load existing network if available

        self.loss_fun = 'rms'
        self.skip_net = False #set it to false when training the network

        # LR scheduler
        self.scheduler = True
        self.scheduler_patience = 5
        self.scheduler_factor = 0.2
        self.min_lr = 1e-6

        # device
        self.use_cuda = torch.cuda.is_available()
        device_str = os.environ.get("HP_DEVICE", "cuda:0" if self.use_cuda else "cpu")
        self.device = torch.device(device_str)

        # reproducibility
        self.select_best = True
        self.seed = int(os.environ.get("HP_SEED", "1337"))


# -------------------------- network params --------------------------
class net_pars:
    def __init__(self, nets):
        self.dropout = 0.10
        self.batch_norm = True
        self.parallel = True

        # IMPORTANT: use 'sigmoid' so FRACTIONS obey min/max (not just abs)
        # ('sigmoidabs' would NOT cap fractions at the upper bound)
        self.con = 'sigmoid'

        # Order: [Dpar, Fint, Dint, Fmv, Dmv, S0]
        # Correct order: [Dpar, Dint, Dmv, Fint, Fmv, S0]
       #self.cons_min = [0.0001, 0, 0.0015, 0, 0.004, 0.80]
        self.cons_min = [0.0001, 0.005, 0.0015, 0.0, 0.004, 0.90]
        self.cons_max = [0.0015, 0.5, 0.005, 0.05, 0.2, 1.10]
      #self.cons_max = [0.0015, 0.004, 0.2, 0.4, 0.05, 1.20]
        boundsrange = 0.3 * (np.array(self.cons_max) - np.array(self.cons_min))
        self.cons_min = (np.array(self.cons_min) - boundsrange).tolist()
        self.cons_max = (np.array(self.cons_max) + boundsrange).tolist()


        # Do NOT widen bounds (keeps fractions from inflating)
        # boundsrange = 0.3 * (np.array(self.cons_max) - np.array(self.cons_min))
        # self.cons_min = (np.array(self.cons_min) - boundsrange).tolist()
        # self.cons_max = (np.array(self.cons_max) + boundsrange).tolist()

        self.fitS0 = True # main script may override to False if you don't want S0
        self.IR = False # True for IR (inversion recovery); False for standard

        self.depth = 2
        self.width = 0        # 0 => width = #b-values


# -------------------------- LSQ baseline --------------------------
class lsqfit:
    def __init__(self):
        self.do_fit = True
        self.fitS0 = True
        self.jobs = int(os.environ.get("HP_LSQ_JOBS", "1"))
        # bounds order: [S0, Dpar, Fint, Dint, Fmv, Dmv]
        self.bounds = (
            [0.95, 0.0003, 0.00,  0.0018, 0.00,  0.070],
            [1.05, 0.0012, 0.25,  0.0035, 0.05,  0.150]
        )


# -------------------------- simulation params --------------------------
class sim:
    def __init__(self):
        # (Used only for synthetic sims; not used when fitting real data)
        #self.bvalues = np.array([0, 30, 90, 210, 280, 350, 580, 620, 660, 680, 720, 760, 980, 990, 1000], dtype=float)
        self.bvalues = np.array([0, 10, 10, 10, 20, 20, 20, 40, 40, 40, 60, 60, 60, 90, 90, 90, 120, 120, 120, 200, 200, 200, 300, 300, 300, 400, 400, 400, 500, 500, 500, 600, 600, 600, 700, 700, 700, 800, 800, 800, 1000, 1000, 1000, 1200, 1200, 1200, 0, 0, 0, 0, 0], dtype=float)

        self.SNR = 60
        self.sims = int(os.environ.get("HP_SIMS", "11500000"))
        self.num_samples_eval = int(os.environ.get("HP_EVAL", "10000"))
        self.distribution = 'uniform'

        self.repeats = 1
        self.n_ensemble = int(os.environ.get("HP_ENSEMBLE", "20"))
        self.jobs = int(os.environ.get("HP_JOBS", "1"))

        self.IR = False # True for IR (inversion recovery); False for standard
        self.rician = True

        # parameter ranges for sims (  # Order: [Dpar, Fint, Dint, Fmv, Dmv])
       #self.range = ( [0.0001, 0.0015, 0.004, 0.0, 0.0 ],  [0.0015, 0.004, 0.2, 0.4, 0.05])
        self.range = ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.70, 0.004, 0.05, 0.20])

        self.chunk_size = int(os.environ.get("HP_CHUNK", "250000"))


# -------------------------- relaxation / timings --------------------------
class rel_times:
    """Relaxation times and acquisition parameters (ms). Used when IR=True."""
    def __init__(self):
        # scanner timings (adjust if your protocol differs)
        self.echotime       = 58       # TE
        self.repetitiontime = 18000     # TR
        self.inversiontime  = 0     # TI (IR/CSF-suppressed) #2200 for IR data 

        # 7T-ish defaults (can be tuned to site-specific values)
      # self.bloodT2  = 23
      # self.tissueT2 = 46
      # self.isfT2    = 100
      # self.bloodT1  = 2596
      # self.tissueT1 = 1900
      # self.isfT1    = 4300

        self.bloodT1  = 2600   # midrange blood T1
        self.bloodT2  = 23     # slightly stricter suppression
        self.tissueT1 = 1200   # closer to GM
        self.tissueT2 = 46     # WM–GM average
        self.isfT1    = 4300   # robust CSF/ISF
        self.isfT2    = 100    # closer to CSF at 7T


# -------------------------- container --------------------------
class hyperparams:
    def __init__(self):
        self.fig = False
        self.save_name = 'brain3'
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = lsqfit()
        self.sim = sim()
        self.rel_times = rel_times()
