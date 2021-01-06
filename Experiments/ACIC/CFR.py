import sys,os
# import time
from time import time

import numpy as np
import tensorflow as tf

# pip install tensorflow_probability-gpu
# import tensorflow_probability as tfp
# tfd = tfp.distributions
# tfb = tfp.bijectors
# layers = tf.contrib.layers

import pandas as pd
import math

import seaborn as sns


import getopt
import random
import datetime
import cfr_net as cfr
import traceback

# pip install progressbar2
# from progressbar import ETA, Bar, Percentage, Progress Bar, Dynamic Message

import matplotlib.pyplot as plt
# %matplotlib inline

# Allow growth to curb resource drain

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
sess = tf.Session(config=config)




#---------------------------------


# ihdp_ind = 0
# ihdp_ind = int(sys.argv[1])



acic_ind = int(sys.argv[1])
folder_ind = (acic_ind//20)+1   # 1-77
file_ind = np.mod(acic_ind,20)    # 0-99



#-----------------------------------------------------


class flags:
    n_in=2 #"""Number of representation layers. """)
    n_out=2 #"""Number of regression layers. """)
    p_alpha=1e-4 #"""Imbalance regularization param. """)
    p_lambda=0.0 #"""Regularization parameter. """)
    dropout_in=1.0 #"""Input layers dropout keep rate. """)
    dropout_out=1.0 #"""Output layers dropout keep rate. """)
    lrate=0.05 #"""Learning rate. """)
    decay=0.5 #"""RMSProp decay. """)
    batch_size=100 #"""Batch size. """)
    dim_in=100 #"""Pre-representation layer dimensions. """)
    dim_out=100 #"""Post-representation layer dimensions. """)
    rbf_sigma=0.1 #"""RBF MMD sigma """)
    experiments=100 #"""Number of experiments. """)
    iterations=2000 #"""Number of iterations. """)
    weight_init=0.01 #"""Weight initialization scale. """)
    outdir='../results/tfnet_topic/alpha_sweep_22_d100/' #"""Output directory. """)
    datapath='../data/topic/csv/' #"""Data directory. """)
    lrate_decay=0.95 #"""Decay of learning rate every 100 iterations """)
    loss='l2' #"""Which loss function to use (l1/l2/log)""")
    seed=1 #"""Seed. """)
    repetitions=1 #"""Repetitions with different seed.""")
    use_p_correction=1 #"""Whether to use population size p(t) in mmd/disc/wass.""")
    varsel=0 #"""Whether the first layer performs variable selection. """)
    imb_fun='wass' #"""Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
    output_csv=1 #"""Whether to save a CSV file with the results""")
    output_delay=200 #"""Number of iterations between outputs. """)
    wass_iterations=20 #"""Number of iterations in Wasserstein computation. """)
    wass_lambda=1 #"""Wasserstein lambda. """)
    wass_bpt=0 #"""Backprop through T matrix? """)
    save_rep=0 #"""Save representations after training. """

FLAGS = flags()
DTYPE = tf.float32

NUM_ITERATIONS_PER_DECAY = 100



VAL = True
NORM =True

HAVE_TRUTH = False


# default setting
#----------------------------------------------------

FLAGS.p_alpha=1e-2
FLAGS.p_lambda=1e-3
FLAGS.n_in=2
FLAGS.n_out=2
FLAGS.dropout_in=1.0
FLAGS.dropout_out=1.0
FLAGS.lrate=0.01
FLAGS.lrate_decay=0.92
FLAGS.decay=0.5
FLAGS.batch_size=100
FLAGS.dim_in=25
FLAGS.dim_out=25
FLAGS.rbf_sigma=0.1
FLAGS.imb_fun='wass'
FLAGS.wass_lambda=1
FLAGS.wass_iterations=10
FLAGS.wass_bpt=1
FLAGS.use_p_correction=1
FLAGS.iterations=2000
FLAGS.weight_init=0.001
FLAGS.outbase='results'
FLAGS.datapath='data/ihdp_sample.csv'
FLAGS.loss='l2'
FLAGS.sparse=0
FLAGS.varsel=0
FLAGS.save_rep=0

FLAGS.dim_rep=25
FLAGS.weight_init = 0.1
FLAGS.iterations=15000
FLAGS.lrate=0.01
FLAGS.output_delay=1000

# modify
#----------------------------------------------------

# FLAGS.dim_rep = 2
# FLAGS.imb_fun='mmd'
# FLAGS.p_alpha = 10. # imbalance regularization parameter
# FLAGS.p_lambda = 0
# FLAGS.dim_in = 25
# FLAGS.dim_out = 25
# FLAGS.weight_init = 0.001
# FLAGS.weight_init = 1.
# FLAGS.imb_fun = 'wass'
# FLAGS.lrate = 1e-3









#----------------------------------------------------------------
#
#    additional FUNCTIONS
#
#----------------------------------------------------------------


def load_ihdp(trial_id=0,filepath='../data/',istrain=True):

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


def load_cfdata(file_dir):
    df = pd.read_csv(file_dir)
    z = df['z'].values
    y0 = df['y0'].values
    y1 = df['y1'].values
    y = y0*(1-z) + y1*z
    return [z,y,df['mu0'].values,df['mu1'].values]






def eval_y_rmse(yy,yy_mdl):

    if len(yy_mdl.shape)>1:
        yy_mdl_mean = np.mean(yy_mdl,axis=1).reshape([-1,1])
    else:
        yy_mdl_mean = yy_mdl

    yy = yy.reshape([-1,1])

    rmse = np.sqrt(np.mean(np.square(yy-yy_mdl_mean)))

    return rmse


eval_pehe = cfr.eval_pehe




def check_results(x_x,t_x,y_x,tau_x,msg=''):

    n_x = x_x.shape[0]

    mu0_pred = sess.run(CFR.output, feed_dict={x_: x_x, t_: np.zeros([n_x,1]), \
        do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda})
    mu1_pred = sess.run(CFR.output, feed_dict={x_: x_x, t_: np.ones([n_x,1]), \
        do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda})
    y_pred = sess.run(CFR.output, feed_dict={x_: x_x, t_: t_x, \
        do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda})
    tau_hat = mu1_pred - mu0_pred
    tau_hat += 1e-2*np.random.randn(n_x,1)
    pehe_ = eval_pehe(tau_hat, tau_x)*y_std
    corr_ = np.corrcoef(tau_hat.reshape([-1,]),tau_x.reshape([-1,]))[0,1]
    err_ = np.mean(tau_hat)-4

    rmse_ = eval_y_rmse(y_x, y_pred)

    print('%sPEHE=%.2f, CORR=%.2f, ERR=%.2f, RMSE=%.2f' % (msg, pehe_, corr_, err_, rmse_) )

    return [pehe_, corr_, err_, rmse_,y_std]


def onehot(t,dim):

    m_samples = t.shape[0]
    tt = np.zeros([m_samples,dim])

    for i in range(m_samples):
        tt[i,np.int(t[i])] = 1

    return tt








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

# folder_ind = 1  # 1-77
folder_dir = './data/acic2016/'+str(folder_ind)+'/'
filelist = os.listdir(folder_dir)

# file_ind = 1  # 0-99
T,Y,Mu0,Mu1 = load_cfdata(folder_dir+filelist[file_ind])


y_mean = np.mean(Y)
y_std = np.std(Y)

Y = (Y-y_mean)/y_std

Mu0 = (Mu0-y_mean)/y_std
Mu1 = (Mu1-y_mean)/y_std

Tau = Mu1 - Mu0
Y = np.reshape(Y,[-1,1])

# T = onehot(T,2)
# Tau.shape
# np.mean(Tau)
# X.shape

n_samples = X.shape[0]



# Testing data set: from row 4000 to n_samples:
#----------------------------------------------------------

test_ind = 4000

X_test = X[4000:,:]
Y_test = Y[4000:,:]
T_test = T[4000:]
Mu0_test = Mu0[4000:]
Mu1_test = Mu1[4000:]


Tau_test = Mu1_test - Mu0_test
Y_test = np.reshape(Y_test,[-1,1])
# T_test = np.reshape(T_test[-1,])

n_samples_test = X_test.shape[0]


# t1_ind_test = T_test[:,1]==1   # find which column has the treatment == 1
# t0_ind_test = T_test[:,0]==1

# n0_test = np.sum(t0_ind_test)
# n1_test = np.sum(t1_ind_test)

# X0_test = X_test[t0_ind_test]
# X1_test = X_test[t1_ind_test]

# Y0_test = Y_test[t0_ind_test]
# Y1_test = Y_test[t1_ind_test]







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




# # data formating
# #----------------
# t1_ind = T[:,1]==1   # find which column has the treatment == 1
# t0_ind = T[:,0]==1

# n0 = np.sum(t0_ind)
# n1 = np.sum(t1_ind)

# X0 = X[t0_ind]
# X1 = X[t1_ind]

# Y0 = Y[t0_ind]
# Y1 = Y[t1_ind]




T = T.reshape([-1,1])
Y = Y.reshape([-1,1])
# y_cf_all = Ycf.reshape([-1,1])
Tau = Tau.reshape([-1,1])

T_val = T_val.reshape([-1,1])
T_test = T_test.reshape([-1,1])

dim = X.shape[1]
n = X.shape[0]






#----------------------------------------------------------------
#
#    MODEL
#
#----------------------------------------------------------------


''' Initialize input placeholders '''
x_  = tf.placeholder("float", shape=[None,dim], name='x_') # Features
t_  = tf.placeholder("float", shape=[None,1], name='t_')   # Treatent
y_ = tf.placeholder("float", shape=[None,1], name='y_')  # Outcome


''' Parameter placeholders '''
alpha_ = tf.placeholder("float", name='alpha_')
lambda_ = tf.placeholder("float", name='lambda_')
do_in = tf.placeholder("float", name='dropout_in')
do_out = tf.placeholder("float", name='dropout_out')
p = tf.placeholder("float", name='p_treated')


''' Define model graph '''
# log(logfile, 'Defining graph...')
dims = [dim,FLAGS.dim_in,FLAGS.dim_out]
CFR = cfr.cfr_net(x_, t_, y_, p, FLAGS, alpha_, lambda_, do_in, do_out, dims)

if FLAGS.varsel:
    w_proj = tf.placeholder("float", shape=[dim], name='w_proj')
    projection = CFR.weights_in[0].assign(w_proj)




''' Set up optimizer '''
# log(logfile, 'Training...')
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
    NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
train_step = tf.train.RMSPropOptimizer(lr, FLAGS.decay).minimize(CFR.tot_loss,global_step=global_step)

''' Compute treatment probability'''
t_cf_all = 1-T
if FLAGS.use_p_correction:
    p_treated = np.mean(T)
else:
    p_treated = 0.5

''' Set up loss feed_dicts'''
dict_factual = {x_: X, t_: T, y_: Y, \
    do_in:1.0, do_out:1.0, alpha_:FLAGS.p_alpha, \
    lambda_:FLAGS.p_lambda, p:p_treated}

if HAVE_TRUTH:
    dict_cfactual = {x_: X, t_: t_cf_all, y_: y_cf_all, \
        do_in:1.0, do_out:1.0}



''' Initialize tensorflow variables '''
sess.run(tf.initialize_all_variables())



''' Compute losses before training'''
losses = []
obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, \
    CFR.imb_loss], feed_dict=dict_factual)

cf_error = np.nan
if HAVE_TRUTH:
    cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

losses.append([obj_loss, f_error, cf_error, imb_err])

# log(logfile, 'Objective Factual CFactual Imbalance')
# log(logfile, str(losses[0]))


epoch = int(FLAGS.iterations/FLAGS.output_delay)
results = np.zeros([epoch, 3, 5 ])
epoch_id = 0

''' Train for m iterations '''
for i in range(FLAGS.iterations):

    ''' Fetch sample '''
    I = random.sample(range(1, n), FLAGS.batch_size)
    x_batch = X[I,:]
    t_batch = T[I]
    y_batch = Y[I]

    ''' Do one step of gradient descent '''
    sess.run(train_step, feed_dict={x_: x_batch, t_: t_batch, \
        y_: y_batch, do_in:FLAGS.dropout_in, do_out:FLAGS.dropout_out, \
        alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda, p:p_treated})

    ''' Project variable selection weights '''
    if FLAGS.varsel:
        wip = cfr.simplex_project(sess.run(CFR.weights_in[0]), 1)
        sess.run(projection,feed_dict={w_proj: wip})

    ''' Compute loss every N iterations '''
    if i % FLAGS.output_delay == 0:
        obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_loss],
            feed_dict=dict_factual)

        y_pred = sess.run(CFR.output, feed_dict={x_: x_batch, t_: t_batch, \
            y_: y_batch, do_in:FLAGS.dropout_in, do_out:FLAGS.dropout_out, \
            alpha_:FLAGS.p_alpha, lambda_:FLAGS.p_lambda, p:p_treated})

        cf_error = np.nan
        if HAVE_TRUTH:
            cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

        losses.append([obj_loss, f_error, cf_error, imb_err])
        loss_str = str(i) + '\tObj: %.4g,\tF: %.4g,\tCf: %.4g,\tImb: %.4g' % (obj_loss, f_error, cf_error, imb_err)

        if FLAGS.loss == 'log':
            y_pred = 1.0*(y_pred>0.5)
            acc = 100*(1-np.mean(np.abs(y_batch-y_pred)))
            loss_str += ',\tAcc: %.2f%%' % acc

        # log(logfile, loss_str)
        print(loss_str)

        # Within-sample
        max_n = 400
        res1 = check_results(X,T,Y,Tau,msg='training-sample:\t')

        #Val
        res2 = check_results(X_val,T_val,Y_val,Tau_val,msg='validation-sample:\t')

        #Out-of-sample
        res3 = check_results(X_test,T_test,Y_test,Tau_test,msg='test-sample:\t')

        results[epoch_id,0,:] = res1
        results[epoch_id,1,:] = res2
        results[epoch_id,2,:] = res3
        epoch_id+=1

#----------------------------------------------------------------
#
#    OUTPUT
#
#----------------------------------------------------------------


np.save(str(folder_ind)+'_'+str(file_ind)+"_results.npy",results)
