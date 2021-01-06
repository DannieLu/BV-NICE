import os
import sys
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

# import seaborn as sns

# pip install progressbar2
# from progressbar import ETA, Bar, Percentage, Progress Bar, Dynamic Message

import matplotlib.pyplot as plt
# %matplotlib inline

# Allow growth to curb resource drain

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
sess = tf.Session(config=config)



#------------------------------------------


ihdp_ind = int(sys.argv[1])
# ihdp_ind = 1


lr = 1e-3
max_epoch = 50
updates_per_epoch = 100




eta_list = np.flipud(1./10**np.array(range(5)))
eta_list = np.concatenate([np.array([0]),eta_list])
lam_list = 1./10**np.array(range(5))
lam_list = np.concatenate([np.array([0]),np.flipud(lam_list), np.array(10**np.array(range(1,3)))])

ihdp_id = np.mod(int(sys.argv[1]),1000)
param_id = int(sys.argv[1])//1000

n_lam = len(lam_list)
n_eta = len(eta_list)

for param_id in range(n_lam*n_eta):
    eta_id = param_id // n_lam
    lam_id = np.mod(param_id,n_lam)

    eta = eta_list[eta_id]
    lam = lam_list[lam_id]










#------------------------------------------

NORM = True

VAL = True

# lam = 1 # imbalance      1e^k k=0:4 ; 0
# lam = 0.

zeta = 1. # match rep dist to some prior dist
# zeta = 0


# eta = .1  # noise level      1e^k k=-2:4
# eta = 0.

alph = 0.
# alph = 1e-5   # representation regularization

kappa = .01  # propensity score weight
# kappa = 0.

RDECOMP = True
# RDECOMP = False






#------------------------------------------



class flags:

    # dim = 2

    x_dim = 25
    y_dim = 1
    t_dim = 2
    M = 10

    hidden_units = 64
    hidden_size = 2

    # hidden_size = 10

    xi_dim = 10

    # optimization
    learning_rate = 1e-3 # Base learning rate
    lr_decay = 0.999995 # Learning rate decay, applied every step of the optimization

    batch_size = 128 # Batch size during training per GPU
    # batch_size = 400


FLAGS = flags()
args = FLAGS

DTYPE = tf.float32



#------------------------------------------




initializer = tf.contrib.layers.xavier_initializer()
# initializer = None

# nonlinearity=tf.nn.elu
nonlinearity=tf.nn.relu



def onehot(t,dim):

    m_samples = t.shape[0]
    tt = np.zeros([m_samples,dim])

    for i in range(m_samples):
        tt[i,np.int(t[i])] = 1

    return tt



def critic(z,x,name):

    hidden_units = FLAGS.hidden_units    # size of hidden units in a layer

    if x is None:
        input_tensor = z;
    else:
        input_tensor = tf.concat([x,z],axis=-1)

    with tf.variable_scope('critic-%s' % name,reuse=tf.AUTO_REUSE):
        h1 = tf.layers.dense(input_tensor,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        h2 = tf.layers.dense(h1,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        # h2 = tf.layers.dense(h2,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        # o = tf.layers.dense(h2,1,activation=None)
        o = tf.layers.dense(h2,1,kernel_initializer=initializer,activation=tf.nn.tanh)
        o *= 5

    return o;

def simple_mlp(x,out_dim,name):

    hidden_units = FLAGS.hidden_units   # size of hidden units in a layer

    input_tensor = x

    with tf.variable_scope('%s' % name,reuse=tf.AUTO_REUSE):
        h1 = tf.layers.dense(input_tensor,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        h2 = tf.layers.dense(h1,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        o = tf.layers.dense(h2,out_dim,kernel_initializer=initializer,activation=None)

    return o;



def encoder(input_x,input_y,input_t,name='encoder'):
    '''
    approximate posterior of z given x,y,t
    p(z|x,y,t)
    return mean and var of z

    '''

    if (input_y is not None) and (input_t is not None):
        input_tensor = tf.concat([input_x,input_y,input_t],axis=-1)  # cbind
    else:
        input_tensor = input_x

    hidden_units = FLAGS.hidden_units   # size of hidden units in a layer

    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        h1 = tf.layers.dense(input_tensor,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        h2 = tf.layers.dense(h1,hidden_units,kernel_initializer=initializer,activation=nonlinearity)
        o = tf.layers.dense(h2,FLAGS.hidden_size,kernel_initializer=initializer,activation=None)
        # o = tf.layers.dense(h2,2*latent_size,activation=None,use_bias=False)

    return o



def estimate_causal_effect(xx, n_runs=1):

    m_samples = xx.shape[0]

    t0 = np.zeros([m_samples,FLAGS.t_dim]); t0[:,0] = 1;
    t1 = np.zeros([m_samples,FLAGS.t_dim]); t1[:,1] = 1;
    mu0_hat = estimate_outcome(xx, t0, n_runs)
    mu1_hat = estimate_outcome(xx, t1, n_runs)

    tau_hat = mu1_hat - mu0_hat

    return tau_hat

def estimate_outcome(xx, tt, n_runs=1):
    m_samples = xx.shape[0]

    y_t_hat = 0

    for i in range(n_runs):

        y_t_hat = tf_eval(y_hat,m_samples,{input_x: xx, input_t: tt})

    y_t_hat /= n_runs

    return y_t_hat




def int_shape(x):
    return list(map(int, x.get_shape()))

def print_shape(x,varname='variable'):
    if x is None:
        print('%s size: None' % (varname))
        return
    x_shape = x.shape.as_list()
    # print('%s size: [%d,%d,%d]' % (varname,x_shape[1],x_shape[2],x_shape[3]))
    print(varname,end=': ')
    print(x_shape)

def tf_eval(tf_tensor,n_samples,feed_dict=None):

    MLOOP = np.int(np.ceil(n_samples/FLAGS.batch_size))

    dd = tf_tensor.shape.as_list()[1:]
    dd.insert(0,n_samples)

    x = np.zeros(dd)

    for mloop in range(MLOOP):

        st = mloop*FLAGS.batch_size
        ed = min((mloop+1)*FLAGS.batch_size, n_samples)

        if feed_dict is not None:
            feed_dict_i = dict()
            for key in feed_dict.keys():
                feed_dict_i[key] = np.random.randn(*int_shape(key))
                feed_dict_i[key][:ed-st] = feed_dict[key][st:ed]
            y = sess.run(tf_tensor,feed_dict=feed_dict_i)
        else:
            y = sess.run(tf_tensor)

        # print([st,ed])
        x[st:ed] = y[:ed-st]

    return x

def tf_eval_list(tf_tensor_list,n_samples,feed_dict=None):

    if isinstance(tf_tensor_list, list)==False:
        print('Input not a list')
        return None

    MLOOP = np.int(np.ceil(n_samples/FLAGS.batch_size))

    res = dict()

    for key in tf_tensor_list:
        dd = key.shape.as_list()[1:]
        dd.insert(0,n_samples)
        res[key] = np.zeros(dd)

    for mloop in range(MLOOP):

        st = mloop*FLAGS.batch_size
        ed = min((mloop+1)*FLAGS.batch_size,n_samples)

        if feed_dict is not None:
            feed_dict_i = dict()
            for key in feed_dict.keys():
                feed_dict_i[key] = np.random.randn(*int_shape(key))
                feed_dict_i[key][:ed-st] = feed_dict[key][st:ed]
            # print(feed_dict_i)
            y = sess.run(tf_tensor_list,feed_dict=feed_dict_i)
        else:
            y = sess.run(tf_tensor_list)

        for i in range(len(tf_tensor_list)):
            res[tf_tensor_list[i]][st:ed] = y[i][:ed-st]

    return res




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


def get_phi_x(input_x):
    xi = tf.random.uniform([FLAGS.batch_size,FLAGS.xi_dim], minval=-1,maxval=1)
    x_xi = tf.concat([input_x,eta*xi],axis=-1)
    phi_x = simple_mlp(x_xi,FLAGS.hidden_size,name='encoder')
    return phi_x





def eval_nll_ps_x(X_val,T_val):

    m_samples = X_val.shape[0]
    ps_val = tf_eval(e_x, m_samples, {input_x: X_val}).reshape([-1,])
    loglik_ps_val = np.log(T_val[:,0]*(1-ps_val)+T_val[:,1]*ps_val)
    ps_nll_val = -np.mean(loglik_ps_val)

    return ps_nll_val



def check_results(x_x,t_x,y_x,tau_x,msg=''):

    tau_hat = estimate_causal_effect(x_x).reshape([-1,])
    pehe_mkl = eval_pehe(tau_hat, tau_x)*y_std
    corr_ = np.corrcoef(tau_hat.reshape([-1,]),tau_x.reshape([-1,]))[0,1]
    err_ = np.mean(tau_hat)-4

    Y_mdl = estimate_outcome(x_x,t_x)
    rmse_ = eval_y_rmse(y_x, Y_mdl)


    print('%sPEHE=%.2f, CORR=%.2f, ERR=%.2f, RMSE=%.2f, y_std=%.2f' % (msg, pehe_mkl, corr_, err_, rmse_,y_std) )

    return [pehe_mkl, corr_, err_, rmse_,y_std]



# def check_results():

#     tau_hat = estimate_causal_effect(X).reshape([-1,])
#     pehe_mkl = eval_pehe(tau_hat, Tau)*y_std
#     print('PEHE = %.2f' % pehe_mkl)

#     return;


def eval_y_rmse(yy,yy_mdl):

    yy_mdl_mean = np.mean(yy_mdl,axis=1).reshape([-1,1])

    rmse = np.sqrt(np.mean(np.square(yy-yy_mdl_mean)))

    return rmse









#----------------------------------------------------------------
#
#     LOAD DATA
#
#----------------------------------------------------------------



# load trainning dataset
#------------------------------------------------

X,Y,T,Ycf,Mu0,Mu1 = load_ihdp(ihdp_ind)
Tau = Mu1 - Mu0

Y = np.reshape(Y,[-1,1])
T = onehot(T,2)

X_m = np.mean(X,axis=0,keepdims=True)
X_std = np.std(X,axis=0,keepdims=True)
X = (X-X_m)/X_std

n_samples = X.shape[0]

y_std = 1.

if NORM:
    y_m = np.mean(Y)
    y_std = np.std(Y)

    Y = (Y-y_m)/y_std

    Tau = Tau/y_std


# split trainning dataset for CV
#-------------------------------

if VAL:

    prob_train = 0.7

    n_train_samples = int(np.ceil(prob_train*n_samples))

    shuff_idx = np.array(range(n_samples))
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




# load testing dataset
#------------------------------------------------

# load testing dataset
X_test,Y_test,T_test,Ycf_test,Mu0_test,Mu1_test = load_ihdp(ihdp_ind,istrain=False)


Tau_test = Mu1_test - Mu0_test
Y_test = np.reshape(Y_test,[-1,1])

if NORM:

    Y_test  = (Y_test -y_m )/y_std
    Tau_test  = Tau_test /y_std


T_test = onehot(T_test,2)
# Tau.shape
# np.mean(Tau)
# X.shape


X_test = (X_test-X_m)/X_std

n_samples_test = X_test.shape[0]


t1_ind_test = T_test[:,1]==1   # find which column has the treatment == 1
t0_ind_test = T_test[:,0]==1

n0_test = np.sum(t0_ind_test)
n1_test = np.sum(t1_ind_test)

X0_test = X_test[t0_ind_test]
X1_test = X_test[t1_ind_test]

Y0_test = Y_test[t0_ind_test]
Y1_test = Y_test[t1_ind_test]










#----------------------------------------------------------------
#
#     MODELS
#
#----------------------------------------------------------------


# encoder
#--------------------------


# Model specification

input_x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.x_dim])
input_t = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.t_dim])
input_y = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.y_dim])

phi_x = get_phi_x(input_x)
# phi_x = simple_mlp(input_x,FLAGS.hidden_size,name='encoder')
# if eta>0: phi_x += eta*tf.random.normal([FLAGS.batch_size, FLAGS.hidden_size])



# Propensity score

logit_t_x = simple_mlp(phi_x, 2, 'predictor_ps') # propensity score
e_x = tf.reshape(tf.nn.softmax(logit_t_x)[:,1],[-1,1])

loss_e_x_vec = tf.nn.softmax_cross_entropy_with_logits(labels=input_t, logits=logit_t_x)
loss_e_x = tf.reduce_mean(loss_e_x_vec)


if RDECOMP:
    # Robinson decomposition

    print('Use Robinson decomposition')

    m_x = simple_mlp(phi_x, 1, 'predictor_m') # expected outcome
    tau_x = simple_mlp(phi_x, 1, 'predictor_tau') # causal effect

    # Continuous outcome
    input_t_bin = tf.reshape(tf.cast(input_t[:,1],dtype=tf.float32),[-1,1])
    y_hat = m_x + (tf.cast(input_t_bin, dtype=tf.float32) - e_x) * tau_x

    # robinson_res = (input_y - m_z) - (tf.cast(input_t_bin, dtype=tf.float32) - e_z) * tau_z
    # loss_y = tf.reduce_mean(tf.square(robinson_res))

else:

    print('Use regular predictor')

    phi_xt = tf.concat([phi_x,tf.cast(input_t,tf.float32)],axis=1)
    y_hat = simple_mlp(phi_xt,FLAGS.y_dim,name='predictor_y')

loss_y = tf.reduce_mean(tf.square(input_y-y_hat))

if alph>0:
    loss_phi = tf.reduce_sum(tf.square(phi_x))
    loss_y += loss_phi

if zeta>0:

    nu_q_vec = critic(phi_x, None, 'ELBO')
    nu_p_vec = critic(tf.random.normal([FLAGS.batch_size, FLAGS.hidden_size]), None, 'ELBO')

    loss_kl = tf.reduce_mean(nu_q_vec) - tf.reduce_mean(tf.exp(nu_p_vec))



input_x_t0 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.x_dim])
input_x_t1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.x_dim])


phi_x_t0 = get_phi_x(input_x_t0)
phi_x_t1 = get_phi_x(input_x_t1)
# phi_x_t0 = simple_mlp(input_x_t0,FLAGS.hidden_size,name='encoder')
# phi_x_t1 = simple_mlp(input_x_t1,FLAGS.hidden_size,name='encoder')

# if eta>0:
#     phi_x_t0 += eta*tf.random.normal([FLAGS.batch_size, FLAGS.hidden_size])
#     phi_x_t1 += eta*tf.random.normal([FLAGS.batch_size, FLAGS.hidden_size])

nu0_vec = critic(phi_x_t0, None, 'KL')
nu1_vec = critic(phi_x_t1, None, 'KL')

loss_fkl = tf.reduce_mean(nu0_vec) - tf.reduce_mean(tf.exp(nu1_vec))

# if zeta>0: loss_fkl += loss_kl


LAM = tf.placeholder(tf.float32)

loss_cfr_kl = loss_y + LAM*loss_fkl
if zeta>0: loss_cfr_kl += zeta*loss_kl
if kappa>0: loss_cfr_kl += loss_e_x



enc_vars = [v for v in tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('encoder')]
dec_vars = [v for v in tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('predictor')]
critic_vars = [v for v in tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('critic')]


learning_rate = tf.placeholder(tf.float32)

Optimizer = tf.train.AdamOptimizer
# Optimizer = tf.train.RMSPropOptimizer


train_cfr_kl = Optimizer(learning_rate).minimize(loss_cfr_kl,var_list=enc_vars+dec_vars)   # for encoder/decoder

kl_list = -loss_fkl
if zeta>0:
    kl_list += -loss_kl
train_critic = Optimizer(learning_rate).minimize(kl_list, var_list=critic_vars)





# Initialization

initializer = tf.global_variables_initializer()
sess.run(initializer)



# Training for encoder/deconder
# lam = 0.01

# lr = 1e-5
# lr = 1e-3
# lr = 1e-4

# max_epoch = 10
# updates_per_epoch = 1000

# max_epoch = 10
# updates_per_epoch = 300

results = np.zeros([max_epoch, 3, 5 ])
epoch_record = np.zeros([max_epoch,])

print(eval_nll_ps_x(X_val,T_val))

for epoch_id in range(max_epoch):

    loss_record = np.zeros([updates_per_epoch,])
    loss_fkl_record = np.zeros([updates_per_epoch,])

    t0 = time()

    for step in range(updates_per_epoch):

        # n_samples
        ind = np.random.choice(n_samples,FLAGS.batch_size)
        # ind
        feed_dict = {learning_rate:lr, LAM:lam}
        feed_dict[input_x] = X[ind]
        feed_dict[input_y] = Y[ind]
        feed_dict[input_t] = T[ind]

        ind = np.random.choice(n0,FLAGS.batch_size)
        feed_dict[input_x_t0] = X0[ind]
        ind = np.random.choice(n1,FLAGS.batch_size)
        feed_dict[input_x_t1] = X1[ind]

        _,_,loss_fkl_val,loss_y_val = sess.run([train_critic, train_cfr_kl,
                                        loss_fkl, loss_y], feed_dict)


        for i in range(0):

            ind = np.random.choice(n0,FLAGS.batch_size)
            feed_dict[input_x_t0] = X0[ind]
            ind = np.random.choice(n1,FLAGS.batch_size)
            feed_dict[input_x_t1] = X1[ind]

            _ = sess.run(train_critic,feed_dict)

        loss_record[step] = loss_y_val + lam*(loss_fkl_val+1.)
        loss_fkl_record[step] = loss_fkl_val

    t1 = time()

    klm = np.mean(loss_fkl_record)+1
    print([epoch_id+1,np.mean(loss_record),klm,t1-t0])
    epoch_record[epoch_id] = np.mean(loss_record)

#     tau_hat = estimate_causal_effect(X).reshape([-1,])
#     pehe_mkl = eval_pehe(tau_hat, Tau)*y_std
#     print('PEHE = %.2f' % pehe_mkl)

#     check_results()
#     print('PS NLL: %.2f' % eval_nll_ps_x(X_val,T_val))


#     print('PEHE=%.2f, CORR=%.2f' % (pehe_bnice, corr_bnice))

    res1 = check_results(X,T,Y,Tau,msg='training-sample:\t')

    # Val
    res2 = check_results(X_val,T_val,Y_val,Tau_val,msg='validation-sample:\t')

    # Out-of-sample
    res3 = check_results(X_test,T_test,Y_test,Tau_test,msg='test-sample:\t')

    results[epoch_id,0,:] = res1
    results[epoch_id,1,:] = res2
    results[epoch_id,2,:] = res3
    epoch_id+=1



# _ = plt.plot(epoch_record)







#----------------------------------------------------------------
#
#     OUTPUTS
#
#----------------------------------------------------------------



np.save(str(ihdp_ind)+"_results.npy",results)
