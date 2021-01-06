import os
import sys
# import time
from time import time

import numpy as np

import pandas as pd
import math

# import seaborn as sns

# pip install progressbar2
# from progressbar import ETA, Bar, Percentage, Progress Bar, Dynamic Message

import matplotlib.pyplot as plt
# %matplotlib inline





ihdp_ind = int(sys.argv[1])




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


NORM = True
VAL = True








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


def simple_mlp(x,out_dim,name):

    hidden_units = 64   # size of hidden units in a layer

    input_tensor = x

    with tf.variable_scope('%s' % name,reuse=tf.AUTO_REUSE):
        h1 = tf.layers.dense(input_tensor,hidden_units,activation=tf.nn.relu)
        h2 = tf.layers.dense(h1,hidden_units,activation=tf.nn.relu)
        o = tf.layers.dense(h2,out_dim,activation=None)

    return o;

def linear_mdl(x,out_dim,name):

    with tf.variable_scope('%s' % name,reuse=tf.AUTO_REUSE):
        o = tf.layers.dense(x,out_dim,activation=None)

    return o;

def eval_pehe(tau_hat,tau):
    return np.sqrt(np.mean(np.square(tau-tau_hat)))


def mu_learner(x,t):

    input_tensor = tf.concat([x,tf.cast(t,tf.float32)],axis=-1)

    mu = simple_mlp(input_tensor,FLAGS.y_dim,'mu')

    return mu

def onehot(t,dim):

    m_samples = t.shape[0]
    tt = np.zeros([m_samples,dim])

    for i in range(m_samples):
        tt[i,np.int(t[i])] = 1

    return tt



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


def get_best_ols(X,Y,X_val,Y_val,X_all,X_val_all,X_test,alpha_list):

    best_rmse = np.inf
    best_mdl = None

    for alpha in alpha_list:
        model = Ridge(alpha=alpha)
        model.fit(X,Y)
        Y_val_hat = model.predict(X_val)
        rmse = np.sqrt(np.mean(np.square(Y_val_hat-Y_val)))
        if rmse<best_rmse:
            best_mdl = model
            best_rmse = rmse

    Y_hat = best_mdl.predict(X_all)
    Y_val_hat = best_mdl.predict(X_val_all)
    Y_test_hat = best_mdl.predict(X_test)

    return Y_hat,Y_val_hat,Y_test_hat


def load_cfdata(file_dir):
    df = pd.read_csv(file_dir)
    z = df['z'].values
    y0 = df['y0'].values
    y1 = df['y1'].values
    y = y0*(1-z) + y1*z
    return [z,y,df['mu0'].values,df['mu1'].values]






#----------------------------------------------------------------
#
#     LOAD DATA: ihdp data
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


t1_ind = T_val[:,1]==1   # find which column has the treatment == 1
t0_ind = T_val[:,0]==1

n0 = np.sum(t0_ind)
n1 = np.sum(t1_ind)

X0_val = X_val[t0_ind]
X1_val = X_val[t1_ind]

Y0_val = Y_val[t0_ind]
Y1_val = Y_val[t1_ind]




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


Y = Y.reshape([-1,])
Y_val = Y_val.reshape([-1,])
Y_test = Y_test.reshape([-1,])

Y0 = Y0.reshape([-1,])
Y1 = Y1.reshape([-1,])

Y0_val = Y0_val.reshape([-1,])
Y1_val = Y1_val.reshape([-1,])

Y0_test = Y0_test.reshape([-1,])
Y1_test = Y1_test.reshape([-1,])





from sklearn.linear_model import Ridge
# >>> import numpy as np
# >>> n_samples, n_features = 10, 5
# >>> rng = np.random.RandomState(0)
# >>> y = rng.randn(n_samples)
# >>> X = rng.randn(n_samples, n_features)
# >>> clf = Ridge(alpha=1.0)
# >>> clf.fit(X, y)
# Ridge()

alpha_list = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3]
# alpha_list = []


mu0_hat,mu0_val_hat,mu0_test_hat = get_best_ols(X0,Y0,X0_val,Y0_val,X,X_val,X_test,alpha_list)
mu1_hat,mu1_val_hat,mu1_test_hat = get_best_ols(X1,Y1,X1_val,Y1_val,X,X_val,X_test,alpha_list)



tau_hat = mu1_hat - mu0_hat
tau_val_hat= mu1_val_hat - mu0_val_hat
tau_test_hat= mu1_test_hat - mu0_test_hat

pehe_ls = eval_pehe(tau_hat, Tau)*y_std
pehe_val_ls = eval_pehe(tau_val_hat, Tau_val)*y_std
pehe_test_ls = eval_pehe(tau_test_hat, Tau_test)*y_std

# print(pehe_ls)
print([pehe_ls,pehe_val_ls,pehe_test_ls])





results = [ihdp_ind ,pehe_ls,pehe_val_ls,pehe_test_ls,y_std]

#----------------------------------------------------------------
#
#     OUTPUT
#
#----------------------------------------------------------------




np.save(str(ihdp_ind)+"_results.npy",results)
