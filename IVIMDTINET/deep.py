"""
June 2024 by Paulien Voorter
paulien.voorter@gmail.com 
https://www.github.com/paulienvoorter

Code is uploaded as part of our paper: 'Diffusion-derived intravoxel-incoherent motion anisotropy relates to cerebrospinal fluid and blood flow'

"""
# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import os
import copy

# Define the neural network.
class Net(nn.Module):
    def __init__(self, bval, bvec, net_pars):
        """
        this defines the Net class which is the network we want to train.
        :param bval: a 1D array with the b-values (.bval file)
        :param bvec: a 2D array containing the gradient directions (.bvec file)
        :param net_pars: an object with network design options with attributes, explained in Kaandorp et al (2020) MRM:
        fitS0 --> Boolean determining whether S0 is fixed to 1 (False) or fitted (True)
        dropout --> Number between 0 and 1 indicating the amount of dropout regularisation
        batch_norm --> Boolean determining whether to use batch normalisation
        cons_min --> [V1-3min, V4-6min, Fpmin, U1-3min, U4-6min, S0min]
        cons_max --> [V1-3max, V4-6max, Fpmax, U1-3max, U4-6max, S0max]
        depth --> integer giving the network depth (number of layers)
        """
        super(Net, self).__init__()
        self.bvec = bvec
        self.bval = bval
        self.net_pars = net_pars
        if self.net_pars.width == 0:
            self.net_pars.width = len(bval)
        # define module lists. If network is not parallel, we can do with 1 list, otherwise we need a list per parameter
        self.fc_layers0 = nn.ModuleList()
        self.fc_layers1 = nn.ModuleList()
        self.fc_layers2 = nn.ModuleList()
        self.fc_layers3 = nn.ModuleList()
        # loop over the layers
        width = len(bval)
        for i in range(self.net_pars.depth):
            # extend with a fully-connected linear layer
            self.fc_layers0.extend([nn.Linear(width, self.net_pars.width)])
            self.fc_layers1.extend([nn.Linear(width, self.net_pars.width)])
            self.fc_layers2.extend([nn.Linear(width, self.net_pars.width)])
            self.fc_layers3.extend([nn.Linear(width, self.net_pars.width)])
            width = self.net_pars.width
            # if desired, add batch normalisation
            if self.net_pars.batch_norm:
                self.fc_layers0.extend([nn.BatchNorm1d(self.net_pars.width)])
                self.fc_layers1.extend([nn.BatchNorm1d(self.net_pars.width)])
                self.fc_layers2.extend([nn.BatchNorm1d(self.net_pars.width)])
                self.fc_layers3.extend([nn.BatchNorm1d(self.net_pars.width)])
            # add ELU units for non-linearity
            self.fc_layers0.extend([nn.ELU()])
            self.fc_layers1.extend([nn.ELU()])
            self.fc_layers2.extend([nn.ELU()])
            self.fc_layers3.extend([nn.ELU()])
            # if dropout is desired, add dropout regularisation
            if self.net_pars.dropout != 0 and i != (self.net_pars.depth - 1):
                self.fc_layers0.extend([nn.Dropout(self.net_pars.dropout)])
                self.fc_layers1.extend([nn.Dropout(self.net_pars.dropout)])
                self.fc_layers2.extend([nn.Dropout(self.net_pars.dropout)])
                self.fc_layers3.extend([nn.Dropout(self.net_pars.dropout)])
        # Final layer yielding output
        self.encoder0 = nn.Sequential(*self.fc_layers0, nn.Linear(self.net_pars.width, 6)) #outputs 6 Cholesky components of D tensor
        self.encoder1 = nn.Sequential(*self.fc_layers1, nn.Linear(self.net_pars.width, 6)) #outputs 6 Cholesky components of D* tensor
        self.encoder2 = nn.Sequential(*self.fc_layers2, nn.Linear(self.net_pars.width, 1)) #outputs 1 scalar parameter: f
        if self.net_pars.fitS0:
            self.encoder3 = nn.Sequential(*self.fc_layers3, nn.Linear(self.net_pars.width, 1)) #outputs S0, in case it is estimated


    def forward(self, X):
        #get constraints
        V123min = self.net_pars.cons_min[0]
        V123max = self.net_pars.cons_max[0]
        V456min= self.net_pars.cons_min[1]
        V456max= self.net_pars.cons_max[1]
        fmin = self.net_pars.cons_min[2] # not used, since we decided to use an abs constraint for f
        fmax = self.net_pars.cons_max[2] # not used, since we decided to use an abs constraint for f
        U123min = self.net_pars.cons_min[3]
        U123max = self.net_pars.cons_max[3]
        U456min= self.net_pars.cons_min[4]
        U456max= self.net_pars.cons_max[4]
        S0min = self.net_pars.cons_min[5]
        S0max = self.net_pars.cons_max[5]

        params0 = self.encoder0(X)
        params1 = self.encoder1(X)
        params2 = self.encoder2(X)
        if self.net_pars.fitS0:
            params3 = self.encoder3(X)

        # applying constraints
        V_xx = V123min + torch.sigmoid(params0[:, 0].unsqueeze(1)) * (V123max - V123min)
        V_xy = V123min + torch.sigmoid(params0[:, 1].unsqueeze(1)) * (V123max - V123min)
        V_xz = V123min + torch.sigmoid(params0[:, 2].unsqueeze(1)) * (V123max - V123min)
        V_yy = V456min + torch.sigmoid(params0[:, 3].unsqueeze(1)) * (V456max - V456min)
        V_zz = V456min + torch.sigmoid(params0[:, 4].unsqueeze(1)) * (V456max - V456min)
        V_yz = V456min + torch.sigmoid(params0[:, 5].unsqueeze(1)) * (V456max - V456min)
        
        U_xx = U123min + torch.sigmoid(params1[:, 0].unsqueeze(1)) * (U123max - U123min)
        U_xy = U123min + torch.sigmoid(params1[:, 1].unsqueeze(1)) * (U123max - U123min)
        U_xz = U123min + torch.sigmoid(params1[:, 2].unsqueeze(1)) * (U123max - U123min)
        U_yy = U456min + torch.sigmoid(params1[:, 3].unsqueeze(1)) * (U456max - U456min)
        U_zz = U456min + torch.sigmoid(params1[:, 4].unsqueeze(1)) * (U456max - U456min)
        U_yz = U456min + torch.sigmoid(params1[:, 5].unsqueeze(1)) * (U456max - U456min)
        
        Fp = torch.abs(params2[:, 0].unsqueeze(1)) 
            
        if self.net_pars.fitS0:
            S0 = S0min + torch.sigmoid(params3[:, 0].unsqueeze(1)) * (S0max - S0min)
            
        #calculating gT*D*g  from Cholesky components of D
        bvecT_Dt_bvec = self.bvec[0,:]*self.bvec[0,:]*V_xx*V_xx \
                +  self.bvec[1,:]*self.bvec[1,:]*(V_xy*V_xy+V_yy*V_yy) \
                +  self.bvec[2,:]*self.bvec[2,:]*(V_xz*V_xz+V_yz*V_yz+V_zz*V_zz) \
                +  2*self.bvec[1,:]*self.bvec[0,:]*(V_xx*V_yy) \
                +  2*self.bvec[1,:]*self.bvec[2,:]*(V_xy*V_yz+V_yy*V_zz) \
                +  2*self.bvec[0,:]*self.bvec[2,:]*(V_xx*V_zz) 
        
        #calculating gT*Dstar*g from Cholesky components of Dstar
        bvecT_Dp_bvec = self.bvec[0,:]*self.bvec[0,:]*U_xx*U_xx \
                +  self.bvec[1,:]*self.bvec[1,:]*(U_xy*U_xy+U_yy*U_yy) \
                +  self.bvec[2,:]*self.bvec[2,:]*(U_xz*U_xz+U_yz*U_yz+U_zz*U_zz) \
                +  2*self.bvec[1,:]*self.bvec[0,:]*(U_xx*U_yy) \
                +  2*self.bvec[1,:]*self.bvec[2,:]*(U_xy*U_yz+U_yy*U_zz) \
                +  2*self.bvec[0,:]*self.bvec[2,:]*(U_xx*U_zz) 
        
                                                                                                                                                              
        # here we estimate X, the signal as function of b-values and directions g given the predicted IVIM parameters. Although
        # this parameter is not interesting for prediction, it is used in the loss function
        X_temp=[]
        if self.net_pars.fitS0:
            X_temp.append(S0 * ((Fp * torch.exp(-self.bval*bvecT_Dp_bvec) + (1-Fp)* torch.exp(-self.bval*bvecT_Dt_bvec))))
        else:
            X_temp.append(((Fp * torch.exp(-self.bval*bvecT_Dp_bvec) + (1-Fp)* torch.exp(-self.bval*bvecT_Dt_bvec))))
        X = torch.cat(X_temp,dim=1)
        X[X>1]=1 #make sure the network keeps optimizing and does not explode e.g., when signal become much larger than 1
        if self.net_pars.fitS0:
            return X, Fp, V_xx, V_xy, V_xz, V_yy, V_zz, V_yz, U_xx, U_xy, U_xz, U_yy, U_zz, U_yz, S0
        else:
            return X, Fp, V_xx, V_xy, V_xz, V_yy, V_zz, V_yz, U_xx, U_xy, U_xz, U_yy, U_zz, U_yz, torch.ones(len(V_xx))
       


def learn_IVIM(X_train, bval, bvec, arg):
    """
    This program builds a IVIM-DTI-NET network and trains it.
    :param X_train: 2D array of IVIM data we use for training. First axis are the voxels and second axis are the signals at the b-value and gradient vector combination
    :param bval: a 1D array with the b-values (.bval file)
    :param bvec: a 2D array containing the gradient directions (.bvec file)
    :param arg: an object with network design options, check hyperparameters.py for options
    :return net: returns a trained network
    """
    torch.backends.cudnn.benchmark = True

    X_train[X_train > 1.5] = 1.5

    # initialising the network of choice using the input argument arg
    bval = torch.FloatTensor(bval[:]).to(arg.train_pars.device)
    bvec = torch.FloatTensor(bvec[:]).to(arg.train_pars.device)
    net = Net(bval, bvec, arg.net_pars).to(arg.train_pars.device)

    # defining the loss function; not explored yet
    if arg.train_pars.loss_fun == 'rms':
        criterion = nn.MSELoss(reduction='mean').to(arg.train_pars.device)
    elif arg.train_pars.loss_fun == 'L1':
        criterion = nn.L1Loss(reduction='mean').to(arg.train_pars.device)

    # splitting data into learning and validation set; subsequently initialising the Dataloaders
    split = int(np.floor(len(X_train) * arg.train_pars.split))
    train_set, val_set = torch.utils.data.random_split(torch.from_numpy(X_train.astype(np.float32)),
                                                       [split, len(X_train) - split])
    # train loader loads the trianing data. We want to shuffle to make sure data order is modified each epoch and different data is selected each epoch.
    trainloader = utils.DataLoader(train_set,
                                   batch_size=arg.train_pars.batch_size,
                                   shuffle=True,
                                   drop_last=True)
    # validation data is loaded here. By not shuffling, we make sure the same data is loaded for validation every time. We can use substantially more data per batch as we are not training.
    inferloader = utils.DataLoader(val_set,
                                   batch_size=32 * arg.train_pars.batch_size,
                                   shuffle=False,
                                   drop_last=True)

    # defining the number of training and validation batches for normalisation later
    totalit = np.min([arg.train_pars.maxit, np.floor(split // arg.train_pars.batch_size)])
    batch_norm2 = np.floor(len(val_set) // (32 * arg.train_pars.batch_size))

    # defining optimiser
    if arg.train_pars.scheduler:
        optimizer, scheduler = load_optimizer(net, arg)
    else:
        optimizer = load_optimizer(net, arg)

    # Initialising parameters
    best = 1e16
    num_bad_epochs = 0
    loss_train = []
    loss_val = []
    prev_lr = 0
    final_model = copy.deepcopy(net.state_dict())
    torch.set_num_threads(1)
    
    ## Train
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        # initialising and resetting parameters
        net.train()
        running_loss_train = 0.
        running_loss_val = 0.
        for i, X_batch in enumerate(tqdm(trainloader, position=0, leave=True, total=totalit), 0):
            if i > totalit:
                # have a maximum number of batches per epoch to ensure regular updates of whether we are improving
                break
            # zero the parameter gradients
            optimizer.zero_grad()
            # put batch on GPU if pressent
            X_batch = X_batch.to(arg.train_pars.device)
            ## forward + backward + optimize
            X_pred, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(X_batch)
            # removing nans and too high/low predictions to prevent overshooting
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # determine loss for batch; note that the loss is determined by the difference between the predicted signal and the actual signal. The loss does not look at Dt-tensor, D*-tensor or Fp.
            loss = criterion(X_pred, X_batch)
            # updating network
            loss.backward()
            optimizer.step()
            # total loss and determine max loss over all batches
            running_loss_train += loss.item()

        # after training, do validation in unseen data without updating gradients
        print('\n validation \n')
        net.eval()
        # validation is always done over all validation data
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            optimizer.zero_grad()
            X_batch = X_batch.to(arg.train_pars.device)
            # do prediction, only look at predicted IVIM signal
            X_pred, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = net(X_batch)
            X_pred[isnan(X_pred)] = 0
            X_pred[X_pred < 0] = 0
            X_pred[X_pred > 3] = 3
            # validation loss
            loss = criterion(X_pred, X_batch)
            running_loss_val += loss.item()
        # scale losses
        running_loss_train = running_loss_train / totalit
        running_loss_val = running_loss_val / batch_norm2
        # save loss history for plot
        loss_train.append(running_loss_train)
        loss_val.append(running_loss_val)
        # as discussed in Kaandorp et al (2020) MRM, LR is important. This approach allows to reduce the LR if we think it is too
        # high, and return to the network state before it went poorly
        if arg.train_pars.scheduler:
            scheduler.step(running_loss_val)
            if optimizer.param_groups[0]['lr'] < prev_lr:
                net.load_state_dict(final_model)
            prev_lr = optimizer.param_groups[0]['lr']
        # print stuff
        print("\nLoss: {loss}, validation_loss: {val_loss}, lr: {lr}".format(loss=running_loss_train,
                                                                             val_loss=running_loss_val,
                                                                             lr=optimizer.param_groups[0]['lr']))
        # early stopping criteria
        if running_loss_val < best:
            print("\n############### Saving good model ###############################")
            final_model = copy.deepcopy(net.state_dict())
            best = running_loss_val
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == arg.train_pars.patience:
                print("\nDone, best val loss: {}".format(best))
                break

    print("Done")
    # save final fits
    if arg.fig:
        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.figure(1)
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
        plt.figure(1)
        plt.gcf()
        plt.savefig('plots/fig_{name}_best-loss_{loss}.png'.format(name=arg.save_name,loss=best))
        plt.close('all')
    # Restore best model
    if arg.train_pars.select_best:
        net.load_state_dict(final_model)
    del trainloader
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return net


def load_optimizer(net, arg):
    if arg.net_pars.fitS0:
        par_list = [{'params': net.encoder0.parameters(), 'lr': arg.train_pars.lr},
                    {'params': net.encoder1.parameters()}, {'params': net.encoder2.parameters()},
                    {'params': net.encoder3.parameters()}]
    else:
        par_list = [{'params': net.encoder0.parameters(), 'lr': arg.train_pars.lr},
                    {'params': net.encoder1.parameters()}, {'params': net.encoder2.parameters()}]
    if arg.train_pars.optim == 'adam':
        optimizer = optim.Adam(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    elif arg.train_pars.optim == 'sgd':
        optimizer = optim.SGD(par_list, lr=arg.train_pars.lr, momentum=0.9, weight_decay=1e-4)
    elif arg.train_pars.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(par_list, lr=arg.train_pars.lr, weight_decay=1e-4)
    if arg.train_pars.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2,
                                                         patience=round(arg.train_pars.patience / 2))
        return optimizer, scheduler
    else:
        return optimizer


def predict_IVIM(data, bval, bvec, net, arg):
    """
    This program takes a trained network and predicts the IVIM parameters from it.
    :param data: 2D array of IVIM data we want to predict the IVIM parameters from. First axis are the voxels and second axis are the signals at the b-value and gradient vector combination
    :param bval: a 1D array with the b-values (.bval file)
    :param bvec: a 2D array containing the gradient directions (.bvec file)
    :param net: the trained IVIM-DTI-NET network
    :param arg: an object with network design options, check hyperparameters.py for options
    :return param: returns the predicted parameters
    """
    # skip nans.
    mylist = isnan(np.mean(data, axis=1))

    # tell net it is used for evaluation
    net.eval()
    
    # initialise parameters and data
    bval = torch.FloatTensor(bval[:]).to(arg.train_pars.device)
    bvec = torch.FloatTensor(bvec[:]).to(arg.train_pars.device)
    net = Net(bval, bvec, arg.net_pars).to(arg.train_pars.device)
    U1 = np.array([])
    U2 = np.array([])
    U3 = np.array([])
    U4 = np.array([])
    U5 = np.array([])
    U6 = np.array([])
    V1 = np.array([])
    V2 = np.array([])
    V3 = np.array([])
    V4 = np.array([])
    V5 = np.array([])
    V6 = np.array([])
    Fp = np.array([])
    S0 = np.array([])
    
    # initialise dataloader. Batch size can be way larger than used in training
    inferloader = utils.DataLoader(torch.from_numpy(data.astype(np.float32)),
                                   batch_size=2056,
                                   shuffle=False,
                                   drop_last=False)
    # start predicting
    with torch.no_grad():
        for i, X_batch in enumerate(tqdm(inferloader, position=0, leave=True), 0):
            X_batch = X_batch.to(arg.train_pars.device)
            # here the signal is predicted. Note that we now are interested in the parameters and no longer in the predicted signal decay.
            _, Fpt, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, U_xxt, U_xyt, U_xzt, U_yyt, U_zzt, U_yzt, S0t = net(X_batch)
            # Quick and dirty solution to deal with networks not predicting S0
            try:
                S0 = np.append(S0, (S0t.cpu()).numpy())
            except:
                S0 = np.append(S0, S0t)
            U1 = np.append(U1, (U_xxt.cpu()).numpy())
            U2 = np.append(U2, (U_xyt.cpu()).numpy())
            U3 = np.append(U3, (U_xzt.cpu()).numpy())
            U4 = np.append(U4, (U_yyt.cpu()).numpy())
            U5 = np.append(U5, (U_yzt.cpu()).numpy())
            U6 = np.append(U6, (U_zzt.cpu()).numpy())
            V1 = np.append(V1, (V_xxt.cpu()).numpy())
            V2 = np.append(V2, (V_xyt.cpu()).numpy())
            V3 = np.append(V3, (V_xzt.cpu()).numpy())
            V4 = np.append(V4, (V_yyt.cpu()).numpy())
            V5 = np.append(V5, (V_yzt.cpu()).numpy())
            V6 = np.append(V6, (V_zzt.cpu()).numpy())
            Fp = np.append(Fp, (Fpt.cpu()).numpy())
      
    # Calculate Dstar tensor and D-tensor from their Cholesky components  
    Dxx = V1**2;
    Dyy = V2**2+V4**2;
    Dzz = V3**2+V5**2+V6**2;
    Dxy = V1*V4;
    Dyz = V2*V5+V4*V6;
    Dxz = V1*V6;   
    
    Dpxx = U1**2;
    Dpyy = U2**2+U4**2;
    Dpzz = U3**2+U5**2+U6**2;
    Dpxy = U1*U4;
    Dpyz = U2*U5+U4*U6;
    Dpxz = U1*U6;  
    
    del inferloader
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    return [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, Dpxx, Dpxy, Dpxz, Dpyy, Dpyz, Dpzz, Fp, S0]

def isnan(x):
    # this program indicates what are NaNs 
    return x != x

