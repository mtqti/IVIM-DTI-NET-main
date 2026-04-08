"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
Adapted January 2022 by Paulien Voorter
Cleaned/aligned 2026 for 7T pipeline compatibility

Changes from original IVIM3brain-NET (Voorter et al., MRM 2023;90:1657-1671):
  - Fixed voxel-index corruption in predict_IVIM (np.delete before sels)
  - Precompute IR scalar constants in Net.__init__ (avoid per-forward recompute)
  - Replaced O(n^2) np.append loop with list + np.concatenate
  - Refactored 6x duplicated parallel encoder construction into loop
  - Configurable max_epochs (was hardcoded 1000)
  - Removed module-level RNG seeding (defer to training script)
  - Use 'Agg' backend by default (headless-safe)
  - Use np.isnan / torch.isnan instead of custom isnan(x) = x != x
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import matplotlib
import os
import copy
import warnings

# Headless-safe backend; override with MPLBACKEND env var if display available
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
from matplotlib import pyplot as plt


# ---------- Canonical parameter names in constraint/output order ----------
# Net constraints use: [Dpar, Fint, Dint, Fmv, Dmv, S0]
# forward() returns:   X, Dpar, Fmv, Dmv, Dint, Fint, S0
# predict_IVIM returns: [Dpar, Fmv, Dmv, Dint, Fint, S0]

# Parallel encoder mapping (encoder index -> parameter name)
_PAR_ENCODER_NAMES = ["Dmv", "Dpar", "Fmv", "Dint", "Fint"]
# With fitS0, encoder5 -> S0


def _build_encoder_layers(depth, in_width, hidden_width, batch_norm, dropout):
    """Build a list of layers for one parallel encoder branch."""
    layers = nn.ModuleList()
    w = in_width
    for i in range(depth):
        layers.append(nn.Linear(w, hidden_width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_width))
        layers.append(nn.ELU())
        if dropout > 0 and i != (depth - 1):
            layers.append(nn.Dropout(dropout))
        w = hidden_width
    return layers


class Net(nn.Module):
    def __init__(self, bvalues, net_pars, rel_times):
        super(Net, self).__init__()
        self.bvalues = bvalues
        self.net_pars = net_pars
        self.rel_times = rel_times

        if self.net_pars.width == 0:
            self.net_pars.width = len(bvalues)

        self.est_pars = 6 if self.net_pars.fitS0 else 5
        in_width = len(bvalues)

        if self.net_pars.parallel:
            # Build one encoder per parameter via loop (was 6x copy-paste)
            n_encoders = 6 if self.net_pars.fitS0 else 5
            self.encoders = nn.ModuleList()
            for _ in range(n_encoders):
                layers = _build_encoder_layers(
                    self.net_pars.depth, in_width, self.net_pars.width,
                    self.net_pars.batch_norm, self.net_pars.dropout,
                )
                self.encoders.append(
                    nn.Sequential(*layers, nn.Linear(self.net_pars.width, 1))
                )
            # Convenience aliases matching original code for optimizer param groups
            self.encoder0 = self.encoders[0]  # Dmv
            self.encoder1 = self.encoders[1]  # Dpar
            self.encoder2 = self.encoders[2]  # Fmv
            self.encoder3 = self.encoders[3]  # Dint
            self.encoder4 = self.encoders[4]  # Fint
            if self.net_pars.fitS0:
                self.encoder5 = self.encoders[5]  # S0
        else:
            layers = _build_encoder_layers(
                self.net_pars.depth, in_width, self.net_pars.width,
                self.net_pars.batch_norm, self.net_pars.dropout,
            )
            self.encoder0 = nn.Sequential(*layers, nn.Linear(self.net_pars.width, self.est_pars))

        # Precompute IR scalar constants (never change during training)
        if self.net_pars.IR:
            rt = self.rel_times
            self._ir_tissue_L = float(
                1 - 2 * np.exp(-rt.inversiontime / rt.tissueT1)
                + np.exp(-rt.repetitiontime / rt.tissueT1)
            )
            self._ir_isf_L = float(
                1 - 2 * np.exp(-rt.inversiontime / rt.isfT1)
                + np.exp(-rt.repetitiontime / rt.isfT1)
            )
            self._ir_blood_L = float(
                1 - np.exp(-rt.repetitiontime / rt.bloodT1)
            )
            self._ir_tissue_T2 = float(np.exp(-rt.echotime / rt.tissueT2))
            self._ir_isf_T2 = float(np.exp(-rt.echotime / rt.isfT2))
            self._ir_blood_T2 = float(np.exp(-rt.echotime / rt.bloodT2))

    def forward(self, X):
        if self.net_pars.con not in ['sigmoid', 'sigmoidabs', 'none', 'abs']:
            raise Exception("constraint must be 'sigmoid', 'sigmoidabs', 'none', or 'abs'")

        if self.net_pars.con in ['sigmoid', 'sigmoidabs']:
            # Canonical order: [Dpar, Fint, Dint, Fmv, Dmv, S0]
            Dparmin, Fintmin, Dintmin, Fmvmin, Dmvmin, S0min = self.net_pars.cons_min
            Dparmax, Fintmax, Dintmax, Fmvmax, Dmvmax, S0max = self.net_pars.cons_max

        if self.net_pars.con == 'abs':
            params0 = torch.abs(self.encoder0(X))
            if self.net_pars.parallel:
                params1 = torch.abs(self.encoder1(X))
                params2 = torch.abs(self.encoder2(X))
                params3 = torch.abs(self.encoder3(X))
                params4 = torch.abs(self.encoder4(X))
                if self.net_pars.fitS0:
                    params5 = torch.abs(self.encoder5(X))
        else:
            params0 = self.encoder0(X)
            if self.net_pars.parallel:
                params1 = self.encoder1(X)
                params2 = self.encoder2(X)
                params3 = self.encoder3(X)
                params4 = self.encoder4(X)
                if self.net_pars.fitS0:
                    params5 = self.encoder5(X)

        if self.net_pars.con == 'sigmoid':
            if self.net_pars.parallel:
                Dmv = Dmvmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv = Fmvmin + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (Fmvmax - Fmvmin)
                Dint = Dintmin + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (Dintmax - Dintmin)
                Fint = Fintmin + torch.sigmoid(params4[:, 0].unsqueeze(1)) * (Fintmax - Fintmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params5[:, 0].unsqueeze(1)) * (S0max - S0min)
            else:
                Dmv = Dmvmin + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv = Fmvmin + torch.sigmoid(params0[:, 2].unsqueeze(1)) * (Fmvmax - Fmvmin)
                Dint = Dintmin + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (Dintmax - Dintmin)
                Fint = Fintmin + torch.sigmoid(params0[:, 4].unsqueeze(1)) * (Fintmax - Fintmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params0[:, 5].unsqueeze(1)) * (S0max - S0min)

        elif self.net_pars.con == 'sigmoidabs':
            # All six parameters now use sigmoid with their physical bounds.
            # Fmv and Fint were previously torch.abs+clamp, which zeros the gradient
            # at the ceiling and causes the network to get stuck at Fmvmax/Fintmax.
            # Sigmoid gives a smooth bounded output in [min, max] with non-zero
            # gradient everywhere, consistent with how Dpar/Dint/Dmv/S0 are handled.
            if self.net_pars.parallel:
                Dmv  = Dmvmin  + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax  - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv  = Fmvmin  + torch.sigmoid(params2[:, 0].unsqueeze(1)) * (Fmvmax  - Fmvmin)
                Dint = Dintmin + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (Dintmax  - Dintmin)
                Fint = Fintmin + torch.sigmoid(params4[:, 0].unsqueeze(1)) * (Fintmax  - Fintmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params5[:, 0].unsqueeze(1)) * (S0max - S0min)
            else:
                Dmv  = Dmvmin  + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (Dmvmax  - Dmvmin)
                Dpar = Dparmin + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (Dparmax - Dparmin)
                Fmv  = Fmvmin  + torch.sigmoid(params0[:, 2].unsqueeze(1)) * (Fmvmax  - Fmvmin)
                Dint = Dintmin + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (Dintmax  - Dintmin)
                Fint = Fintmin + torch.sigmoid(params0[:, 4].unsqueeze(1)) * (Fintmax  - Fintmin)
                if self.net_pars.fitS0:
                    S0 = S0min + torch.sigmoid(params0[:, 5].unsqueeze(1)) * (S0max - S0min)

        else:  # none/abs
            if self.net_pars.parallel:
                Dmv = params0[:, 0].unsqueeze(1)
                Dpar = params1[:, 0].unsqueeze(1)
                Fmv = params2[:, 0].unsqueeze(1)
                Dint = params3[:, 0].unsqueeze(1)
                Fint = params4[:, 0].unsqueeze(1)
                if self.net_pars.fitS0:
                    S0 = params5[:, 0].unsqueeze(1)
            else:
                Dmv = params0[:, 0].unsqueeze(1)
                Dpar = params0[:, 1].unsqueeze(1)
                Fmv = params0[:, 2].unsqueeze(1)
                Dint = params0[:, 3].unsqueeze(1)
                Fint = params0[:, 4].unsqueeze(1)
                if self.net_pars.fitS0:
                    S0 = params0[:, 5].unsqueeze(1)

        if self.net_pars.IR:
            # Use precomputed scalar constants instead of recomputing per call
            tL = self._ir_tissue_L
            iL = self._ir_isf_L
            bL = self._ir_blood_L
            tT2 = self._ir_tissue_T2
            iT2 = self._ir_isf_T2
            bT2 = self._ir_blood_T2

            _ir_num = (
                (1 - Fmv - Fint) * tL * tT2 * torch.exp(-self.bvalues * Dpar)
                + Fint * iL * iT2 * torch.exp(-self.bvalues * Dint)
                + Fmv * bL * bT2 * torch.exp(-self.bvalues * Dmv)
            )
            _ir_den = torch.clamp(
                (1 - Fmv - Fint) * tL * tT2
                + Fint * iL * iT2
                + Fmv * bL * bT2,
                min=1e-6,
            )
            X = _ir_num / _ir_den
            if self.net_pars.fitS0:
                X = S0 * X
        else:
            X = (1 - Fmv - Fint) * torch.exp(-self.bvalues * Dpar) + Fint * torch.exp(-self.bvalues * Dint) + Fmv * torch.exp(-self.bvalues * Dmv)
            if self.net_pars.fitS0:
                X = S0 * X

        if self.net_pars.fitS0:
            return X, Dpar, Fmv, Dmv, Dint, Fint, S0
        return X, Dpar, Fmv, Dmv, Dint, Fint, torch.ones((len(Dpar), 1), device=Dpar.device)


def learn_IVIM(X_train, bvalues, arg, net=None):
    torch.backends.cudnn.benchmark = True
    arg = checkarg(arg)

    S0 = np.mean(X_train[:, bvalues == 0], axis=1).astype(np.float32)
    S0[S0 == 0] = 1.0
    X_train = X_train / S0[:, None]
    X_train = X_train[~np.isnan(np.mean(X_train, axis=1))]

    X_train = X_train[np.percentile(X_train[:, bvalues < 50], 95, axis=1) < 1.2]
    X_train = X_train[np.percentile(X_train[:, bvalues > 150], 95, axis=1) < 1.0]
    X_train[X_train > 1.5] = 1.5

    # keep numpy bvals for plotting, torch bvals for model
    bvalues_np = np.array(bvalues, dtype=np.float32)
    bvalues_t = torch.FloatTensor(bvalues_np).to(arg.train_pars.device)

    if net is None:
        net = Net(bvalues_t, arg.net_pars, arg.rel_times).to(arg.train_pars.device)
    else:
        net.to(arg.train_pars.device)

    if arg.train_pars.loss_fun == 'rms':
        criterion = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    else:
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device)

    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(
        torch.from_numpy(X_train.astype(np.float32)),
        [split, len(X_train) - split]
    )

    trainloader = utils.DataLoader(
        train_set, batch_size=arg.train_pars.batch_size, shuffle=True, drop_last=True
    )
    inferloader = utils.DataLoader(
        val_set, batch_size=32 * arg.train_pars.batch_size, shuffle=False, drop_last=True
    )

    totalit = int(np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)]))
    batch_norm2 = max(1, int(np.floor(len(val_set) // (32 * arg.train_pars.batch_size))))

    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    best = 1e16
    num_bad_epochs = 0
    prev_lr = 0.0
    final_model = copy.deepcopy(net.state_dict())

    loss_train = []
    loss_val = []

    # Configurable max epochs (was hardcoded 1000)
    max_epochs = getattr(arg.train_pars, 'max_epochs', 1000)

    for epoch in range(max_epochs):
        net.train()
        running_loss_train = 0.0
        running_loss_val = 0.0

        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                break
            optimizer.zero_grad()
            X_batch = X_batch.to(arg.train_pars.device)
            X_pred, *_ = net(X_batch)
            X_pred[torch.isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            loss = criterion(X_pred, X_batch)
            loss.backward()
            optimizer.step()
            running_loss_train += loss.item()

        net.eval()
        with torch.no_grad():
            for _, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
                X_batch = X_batch.to(arg.train_pars.device)
                X_pred, *_ = net(X_batch)
                X_pred[torch.isnan(X_pred)] = 0
                X_pred[X_pred < 0] = 0
                X_pred[X_pred > 3] = 3
                loss = criterion(X_pred, X_batch)
                running_loss_val += loss.item()

        running_loss_train = running_loss_train / max(1, totalit)
        running_loss_val = running_loss_val / max(1, batch_norm2)

        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)

        if epoch > 0:
            plot_progress(
                X_batch.detach().cpu(),
                X_pred.detach().cpu(),
                bvalues_np,
                loss_train,
                loss_val,
                arg
            )

        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']

        if running_loss_val < best:
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            if num_bad_epochs == arg.train_pars.patience:
                break

    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/{name}_fig_fit.png'.format(name=arg.save_name))
        plt.figure(2)
        plt.gcf()
        plt.savefig('plots/{name}_fig_train.png'.format(name=arg.save_name))
        plt.close('all')

    if arg.train_pars.select_best:
        net.load_state_dict(final_model)

    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return net


def load_optimizer(net, arg):
    if arg.net_pars.parallel:
        par_list = [
            {'params': net.encoder0.parameters(), 'lr': arg.train_pars.lr},
            {'params': net.encoder1.parameters()},
            {'params': net.encoder2.parameters()},
            {'params': net.encoder3.parameters()},
            {'params': net.encoder4.parameters()},
        ]
        if arg.net_pars.fitS0:
            par_list.append({'params': net.encoder5.parameters()})
    else:
        par_list = [{'params': net.encoder0.parameters()}]

    if arg.train_pars.optim == 'adam':
        optimizer = optim.Adam(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    elif arg.train_pars.optim == 'sgd':
        optimizer = optim.SGD(par_list, lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adagrad(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)

    if arg.train_pars.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.2, patience=round(arg.train_pars.patience / 2)
        )
        return optimizer, scheduler
    return optimizer


def predict_IVIM(data, bvalues, net, arg):
    """
    Run trained PINN inference.

    FIXED: original code used np.delete (which shrinks array) then computed
    sels on the shrunk array, causing voxel index misalignment.  Now we
    compute a single boolean mask on the original data and apply it once.

    Returns: [Dpar, Fmv, Dmv, Dint, Fint, S0]  (each shape [N_original,])
    """
    arg = checkarg(arg)

    S0 = np.mean(data[:, bvalues == 0], axis=1).astype(np.float32)
    S0[~np.isfinite(S0)] = 1.0
    S0[S0 == 0] = 1.0
    data = data / S0[:, None]

    # --- Build a single combined boolean mask on the original array ---
    lend = len(data)
    sels = np.isfinite(np.mean(data, axis=1))  # no NaNs
    # Percentile filters (only valid rows)
    p95_low = np.full(lend, np.inf)
    p95_high = np.full(lend, np.inf)
    low_b_cols = bvalues < 50
    high_b_cols = bvalues > 150
    if np.any(low_b_cols):
        p95_low[sels] = np.percentile(data[sels][:, low_b_cols], 95, axis=1)
    if np.any(high_b_cols):
        p95_high[sels] = np.percentile(data[sels][:, high_b_cols], 95, axis=1)
    sels = sels & (p95_low < 1.2) & (p95_high < 1.0)

    data_sel = data[sels]

    net.eval()

    # Collect batches in lists, concatenate once (was O(n^2) np.append)
    Dpar_list = []
    Fmv_list = []
    Dmv_list = []
    Dint_list = []
    Fint_list = []
    S0p_list = []

    inferloader = utils.DataLoader(
        torch.from_numpy(data_sel.astype(np.float32)),
        batch_size=2056,
        shuffle=False,
        drop_last=False
    )

    with torch.no_grad():
        for _, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to(arg.train_pars.device)
            _, Dpart, Fmvt, Dmvt, Dintt, Fintt, S0t = net(X_batch)

            S0p_list.append(S0t.cpu().numpy().ravel())
            Dmv_list.append(Dmvt.cpu().numpy().ravel())
            Dpar_list.append(Dpart.cpu().numpy().ravel())
            Fmv_list.append(Fmvt.cpu().numpy().ravel())
            Dint_list.append(Dintt.cpu().numpy().ravel())
            Fint_list.append(Fintt.cpu().numpy().ravel())

    if len(Dpar_list) > 0:
        Dpar = np.concatenate(Dpar_list)
        Fmv = np.concatenate(Fmv_list)
        Dmv = np.concatenate(Dmv_list)
        Dint = np.concatenate(Dint_list)
        Fint = np.concatenate(Fint_list)
        S0p = np.concatenate(S0p_list)
    else:
        Dpar = Fmv = Dmv = Dint = Fint = S0p = np.array([])

    # KEEP strict channel semantics from forward(); DO NOT mean-swap channels.
    if len(Dmv) > 0:
        print(
            "[predict_IVIM] means/stds | "
            f"Dpar {np.mean(Dpar):.6g}/{np.std(Dpar):.6g}, "
            f"Fmv {np.mean(Fmv):.6g}/{np.std(Fmv):.6g}, "
            f"Dmv {np.mean(Dmv):.6g}/{np.std(Dmv):.6g}, "
            f"Dint {np.mean(Dint):.6g}/{np.std(Dint):.6g}, "
            f"Fint {np.mean(Fint):.6g}/{np.std(Fint):.6g}, "
            f"S0 {np.mean(S0p):.6g}/{np.std(S0p):.6g}"
        )

    # Write back into full-length arrays using the original boolean mask
    Dmvtrue = np.zeros(lend)
    Dpartrue = np.zeros(lend)
    Fmvtrue = np.zeros(lend)
    Dinttrue = np.zeros(lend)
    Finttrue = np.zeros(lend)
    S0true = np.zeros(lend)

    Dmvtrue[sels] = Dmv
    Dpartrue[sels] = Dpar
    Fmvtrue[sels] = Fmv
    Dinttrue[sels] = Dint
    Finttrue[sels] = Fint
    S0true[sels] = S0p

    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()

    # Output order used by rest of pipeline:
    # [Dpar, Fmv, Dmv, Dint, Fint, S0]
    return [Dpartrue, Fmvtrue, Dmvtrue, Dinttrue, Finttrue, S0true]


def plot_progress(X_batch, X_pred, bvalues, loss_train, loss_val, arg):
    if not arg.fig:
        return

    inds1 = np.argsort(bvalues)
    X_batch = X_batch[:, inds1]
    X_pred = X_pred[:, inds1]
    bvalues = bvalues[inds1]

    plt.close('all')
    fig, axs = plt.subplots(2, 2)
    for idx, ax in enumerate(axs.flat):
        if idx >= len(X_batch):
            break
        ax.plot(bvalues, X_batch.data[idx], 'o')
        ax.plot(bvalues, X_pred.data[idx])
        ax.set_ylim(min(X_batch.data[idx]) - 0.3, 1.2 * max(X_batch.data[idx]))

    plt.legend(('data', 'estimate from network'))
    for ax in axs.flat:
        ax.set(xlabel='b-value (s/mm2)', ylabel='normalised signal')
        ax.label_outer()

    plt.ion()
    plt.show()
    plt.pause(0.001)

    plt.figure(2)
    plt.clf()
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.yscale("log")
    plt.xlabel('epoch #')
    plt.ylabel('loss')
    plt.legend(('training loss', 'validation loss (after training epoch)'))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def checkarg(arg):
    if not hasattr(arg, 'fig'):
        arg.fig = False
        warnings.warn('arg.fig not defined. Using default of False')
    if not hasattr(arg, 'net_pars'):
        raise ValueError('arg.net_pars missing')
    if not hasattr(arg, 'train_pars'):
        raise ValueError('arg.train_pars missing')
    if not hasattr(arg, 'sim'):
        raise ValueError('arg.sim missing')
    if not hasattr(arg, 'fit'):
        raise ValueError('arg.fit missing')
    if not hasattr(arg, 'rel_times'):
        raise ValueError('arg.rel_times missing')
    return arg