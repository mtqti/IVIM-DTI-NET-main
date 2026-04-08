"""
Hyperparameters for 3C-IVIM PINN (7T)

"""

import os
import torch
import numpy as np


class train_pars:
    def __init__(self, nets):
        self.optim = 'adam'
        self.lr = 3e-5
        self.patience = 10
        self.batch_size = 128
        self.maxit = 1000
        self.max_epochs = int(os.environ.get("HP_MAX_EPOCHS", "1000"))
        self.split = 0.90
        self.load_nn = False
        self.loss_fun = 'rms'
        self.skip_net = False

        self.scheduler = True
        self.scheduler_patience = 5
        self.scheduler_factor = 0.2
        self.min_lr = 1e-6

        self.use_cuda = torch.cuda.is_available()
        device_str = os.environ.get("HP_DEVICE", "cuda:0" if self.use_cuda else "cpu")
        self.device = torch.device(device_str)

        self.select_best = True
        self.seed = int(os.environ.get("HP_SEED", "1337"))


class net_pars:
    def __init__(self, nets):
        self.dropout = 0.10
        self.batch_norm = True
        self.parallel = True

        
        self.con = 'sigmoidabs'

        
        self.cons_min = [0.0001, 0.005, 0.0015, 0.0, 0.004, 0.90]
        self.cons_max = [0.0015, 0.4,   0.004,  0.05, 0.2,   1.10]

        self.fitS0 = True
        self.IR = False
        self.depth = 2
        self.width = 0


class lsqfit:
    def __init__(self):
        self.do_fit = True
        self.fitS0 = True
        self.jobs = int(os.environ.get("HP_LSQ_JOBS", "1"))
        
        self.bounds = (
            [0.95, 0.0003, 0.00,  0.0018, 0.00,  0.070],
            [1.05, 0.0012, 0.25,  0.0035, 0.05,  0.150]
        )


class sim:
    def __init__(self):
        self.bvalues = np.array(
            [0, 10, 10, 10, 20, 20, 20, 40, 40, 40, 60, 60, 60, 90, 90, 90, 120, 120, 120,
             200, 200, 200, 300, 300, 300, 400, 400, 400, 500, 500, 500, 600, 600, 600,
             700, 700, 700, 800, 800, 800, 1000, 1000, 1000, 1200, 1200, 1200, 0, 0, 0, 0, 0],
            dtype=float
        )

      
        self.snr_min = 30
        self.snr_max = 60
        self.sims = int(os.environ.get("HP_SIMS", "11500000"))
        self.num_samples_eval = int(os.environ.get("HP_EVAL", "10000"))
        self.distribution = 'uniform'

        self.repeats = 1
        
        self.n_ensemble = int(os.environ.get("HP_ENSEMBLE", "20"))
        self.jobs = int(os.environ.get("HP_JOBS", "1"))

        self.IR = False
        self.rician = True

        # FINAL agreed sim range [Dpar, Fint, Dint, Fmv, Dmv]
        self.range = ([0.0001, 0.005, 0.0015, 0.0, 0.004],
                      [0.0015, 0.4,   0.004,  0.05, 0.20])

        self.chunk_size = int(os.environ.get("HP_CHUNK", "250000"))


class rel_times:
    def __init__(self):
        self.echotime = 58
        self.repetitiontime = 18000
        self.inversiontime = 0

        # Blood T1 at 7T: 2100 ms
        # Li W, Grgac K, Huang A, Yadav N, Qin Q, van Zijl PC.
        # "Quantitative theory for the longitudinal relaxation time of blood water."
        # Magn Reson Med. 2016;76(1):270-81.
        # Rooney WD et al. "Magnetic field and tissue dependencies of human brain
        # longitudinal 1H2O relaxation in vivo." Magn Reson Med. 2007;57(2):308-18.
        # Arterial T1 = 2290+/-100 ms; in vivo venous = 2450+/-110 ms at 7T.
        # 2100 ms used as conservative estimate for mixed arterial/capillary blood.
        self.bloodT1 = 2300

        # Blood T2 at 7T: 23 ms
        # Grgac K, Li W, Huang A, Qin Q, van Zijl PCM.
        # "Transverse water relaxation in whole blood and erythrocytes at 3T, 7T,
        #  9.4T, 11.7T and 16.4T." Magn Reson Imaging. 2017;38:234-249.
        # Krishnamurthy LC et al. "Dependence of blood T2 on oxygenation at 7T."
        # Magn Reson Med. 2014;71:2035-42.
        # T2blood at 7T = 15-25 ms depending on Hct and oxygenation.
        # 23 ms appropriate for normal oxygenation and Hct ~0.44. Unchanged.
        self.bloodT2 = 23

        # Tissue T1 at 7T: 1500 ms (weighted WM/GM average)
        # Wright PJ et al. "Water proton T1 measurements in brain tissue at 7, 3,
        #  and 1.5T using IR-EPI, IR-TSE, and MPRAGE." MAGMA. 2008;21(1-2):121-30.
        # WM T1 = 1126+/-97 ms; GM T1 = 1939+/-149 ms at 7T.
        # WM is 1126 ms and GM is 1939 ms. A simple midpoint is ~1530 ms. 
        # However, in brain IVIM the most physiologically interesting regions are cortical GM and subcortical structures,
        # which are more vascular and have longer T1. A slight upward revision to 1600 ms would better represent the tissue mix in a typical IVIM acquisition that prioritises grey matter coverage.
        self.tissueT1 = 1700

        # Tissue T2 at 7T: 46 ms (whole-brain average)
        # Marjanska M et al. "Localized 1H NMR spectroscopy in different regions
        #  of human brain in vivo at 7T." NMR Biomed. 2012;25(2):332-9.
        # WM T2 ~35-40 ms; GM T2 ~50-55 ms at 7T.
        # 46 ms is acceptable whole-brain average. Unchanged.
        self.tissueT2 = 46

        # ISF T1 at 7T: 4300 ms
        # Rooney WD et al. (as above).
        # O'Reilly T et al. "In vivo T1 and T2 relaxation time maps of brain tissue."
        # Magn Reson Med. 2022.
        # CSF T1 shows no significant B0 dependence, ~4400 ms from 0.15T to 7T.
        # 4300 ms retained as appropriate estimate for ISF. Unchanged.
        self.isfT1 = 4300

        # ISF T2 at 7T: 500 ms
        # Spijkerman JM et al. "T2 mapping of cerebrospinal fluid: 3T versus 7T."
        # MAGMA. 2017;31(3):415-424. Pure CSF T2 at 7T ~1400 ms.
        # ISF in brain parenchyma has higher protein content than ventricular CSF,
        # substantially shortening T2 relative to pure CSF.
        # 500 ms is a conservative estimate for parenchymal ISF T2 at 7T.
        self.isfT2 = 200


class hyperparams:
    def __init__(self):
        self.fig = False
        self.save_name = 'brain3'
        self.net_pars = net_pars(self.save_name)
        self.train_pars = train_pars(self.save_name)
        self.fit = lsqfit()
        self.sim = sim()
        self.rel_times = rel_times()