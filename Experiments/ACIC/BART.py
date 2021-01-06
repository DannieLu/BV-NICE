import os
import sys
# import time
from time import time

import math

import numpy as np

# import seaborn as sns

# pip install progressbar2
# from progressbar import ETA, Bar, Percentage, Progress Bar, Dynamic Message

import matplotlib.pyplot as plt
# %matplotlib inline

from bartpy.sklearnmodel import SklearnModel

import pandas as pd




acic_ind = int(sys.argv[1])
folder_ind = (acic_ind//100)+1   # 1-77
file_ind = np.mod(acic_ind,100)    # 0-99


n_trees = 100 # default is 200 trees



class flags:

    # dim = 2

    x_dim = 25
    y_dim = 1
    t_dim = 2
    # M = 100
    M = 30

    # optimization
    learning_rate = 1e-3 # Base learning rate
    lr_decay = 0.999995 # Learning rate decay, applied every step of the optimization

    batch_size = 128 # Batch size during training per GPU
    hidden_size = 2


FLAGS = flags()
args = FLAGS




def onehot(t,dim):

    m_samples = t.shape[0]
    tt = np.zeros([m_samples,dim])

    for i in range(m_samples):
        tt[i,np.int(t[i])] = 1

    return tt


def eval_pehe(tau_hat,tau):
    return np.sqrt(np.mean(np.square(tau-tau_hat)))


def load_ihdp(trial_id=0,filepath='./data/',istrain=True):

    if istrain:
        data_file = filepath+'ihdp_npci_1-1000.train.npz'
    else:
        data_file = filepath+'ihdp_npci_1-1000.test.npz'

    data = np.load(data_file)

    x = data['x'][:,:,trial_id]
    y = data['yf'][:,trial_id]
    t = data['t'][:,trial_id]
    ycf = data['ycf'][:,trial_id]
    mu0 = data['mu0'][:,trial_id]
    mu1 = data['mu1'][:,trial_id]

    return x,y,t,ycf,mu0,mu1


def show_results():

    return ;


def load_cfdata(file_dir):
    df = pd.read_csv(file_dir)
    z = df['z'].values
    y0 = df['y0'].values
    y1 = df['y1'].values
    y = y0*(1-z) + y1*z
    return [z,y,df['mu0'].values,df['mu1'].values]



#----------------------------------------------------------------
#
#     LOAD DATA : cf data
#
#----------------------------------------------------------------


#


# load covariates

X = pd.read_csv('./data/acic2016/x.csv')
del X['x_2']
del X['x_21']
del X['x_24']

X = X.dropna()

X = X.values

# X = np.delete(X,1,axis=1)  # delete 2nd column

# print(X.head())
X.shape    # (4802, 58)

X_m = np.mean(X,axis=1,keepdims=True)
X_std = np.std(X,axis=1,keepdims=True)
X = (X-X_m)/X_std


# load y and simulations


folder_dir = './data/acic2016/'+str(folder_ind)+'/'
filelist = os.listdir(folder_dir)


T,Y,Mu0,Mu1 = load_cfdata(folder_dir+filelist[file_ind])


y_mean = np.mean(Y)
y_std = np.std(Y)

Y = (Y-y_mean)/y_std

Mu0 = (Mu0-y_mean)/y_std
Mu1 = (Mu1-y_mean)/y_std

Tau = Mu1 - Mu0
Y = np.reshape(Y,[-1,1])

T = onehot(T,2)
# Tau.shape
# np.mean(Tau)
# X.shape

n_samples = X.shape[0]



# Testing data set: from row 4000 to n_samples:
#----------------------------------------------------------

test_ind = 4000

X_test = X[4000:,:]
Y_test = Y[4000:,:]
T_test = T[4000:,:]
Mu0_test = Mu0[4000:]
Mu1_test = Mu1[4000:]


Tau_test = Mu1_test - Mu0_test
Y_test = np.reshape(Y_test,[-1,1])

n_samples_test = X_test.shape[0]


t1_ind_test = T_test[:,1]==1   # find which column has the treatment == 1
t0_ind_test = T_test[:,0]==1

n0_test = np.sum(t0_ind_test)
n1_test = np.sum(t1_ind_test)

X0_test = X_test[t0_ind_test]
X1_test = X_test[t1_ind_test]

Y0_test = Y_test[t0_ind_test]
Y1_test = Y_test[t1_ind_test]




Y0_test = Y0_test.reshape([-1,])
Y1_test = Y1_test.reshape([-1,])




# Validation set
#----------------------------------------------------------


# split 0-4000 rows dataset for CV
#-------------------------------

if VAL:

    prob_train = 0.7

    n_train_samples = int(np.ceil(prob_train*test_ind))

    shuff_idx = np.array(range(test_ind))
    # Shuffle the indices
    # np.random.shuffle(shuff_idx)

    train_idx = shuff_idx[:n_train_samples]
    val_idx = shuff_idx[n_train_samples:]

    X_val = X[val_idx]
    Y_val = Y[val_idx]
    T_val = T[val_idx]

    X = X[train_idx]
    Y = Y[train_idx]
    T = T[train_idx]

    n_samples = n_train_samples

    Tau_val = Tau[val_idx]
    Tau = Tau[train_idx]




# data formating
#----------------
t1_ind = T[:,1]==1   # find which column has the treatment == 1
t0_ind = T[:,0]==1

n0 = np.sum(t0_ind)
n1 = np.sum(t1_ind)

X0 = X[t0_ind]
X1 = X[t1_ind]

Y0 = Y[t0_ind]
Y1 = Y[t1_ind]





Y0 = Y0.reshape([-1,])
Y1 = Y1.reshape([-1,])





#----------------------------------------------------------------
#
#     MODELS
#
#----------------------------------------------------------------




model0 = SklearnModel(n_trees=n_trees) # Use default parameters
model0.fit(X0, Y0) # Fit the model
model1 = SklearnModel(n_trees=n_trees) # Use default parameters
model1.fit(X1, Y1) # Fit the model


tau_hat = model1.predict(X) - model0.predict(X)
tau_hat_val = model1.predict(X_val) - model0.predict(X_val)
tau_hat_test = model1.predict(X_test) - model0.predict(X_test)

pehe_ = eval_pehe(tau_hat, Tau)*y_std
pehe_val = eval_pehe(tau_hat_val, Tau_val)*y_std
pehe_test = eval_pehe(tau_hat_test, Tau_test)*y_std

print(pehe_)
print(pehe_val)
print(pehe_test)

results = [ihdp_ind,pehe_,pehe_val,pehe_test]

#----------------------------------------------------------------
#
#     OUTPUT
#
#----------------------------------------------------------------



np.save(str(folder_ind)+'_'+str(file_ind)+"_results.npy",results)
