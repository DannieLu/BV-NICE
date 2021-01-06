import os
import sys
# import time
from time import time

import numpy as np
import tensorflow as tf

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

DTYPE = tf.float32


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor




NORM = True
VAL = True


ihdp_ind = int(sys.argv[1])



# folder_ind = 1  # 1-77
# file_ind = 1  # 0-99



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


def onehot(t,dim):

    m_samples = t.shape[0]
    tt = np.zeros([m_samples,dim])

    for i in range(m_samples):
        tt[i,np.int(t[i])] = 1

    return tt


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



def mu_learner(x,t):

    input_tensor = tf.concat([x,tf.cast(t,tf.float32)],axis=-1)

    mu = simple_mlp(input_tensor,FLAGS.y_dim,'mu')

    return mu


# def estimate_causal_effect(xx, e_x_, m_x_, n_runs=1):

#     m_samples = xx.shape[0]

#     t0 = np.zeros([m_samples,FLAGS.t_dim]); t0[:,0] = 1;
#     t1 = np.zeros([m_samples,FLAGS.t_dim]); t1[:,1] = 1;
#     mu0_hat = estimate_outcome(xx, t0, e_x_, m_x_, n_runs)
#     mu1_hat = estimate_outcome(xx, t1, e_x_, m_x_, n_runs)

#     tau_hat = mu1_hat - mu0_hat

#     return tau_hat


def estimate_causal_effect(model,X):

    n = X.shape[0]

    t0 = np.zeros([n,1])
    t1 = np.ones([n,1])

    Xt0 = np.concatenate([X,t0],axis=1)
    Xt1 = np.concatenate([X,t1],axis=1)

    mu0_hat = model.predict(Xt0)
    mu1_hat = model.predict(Xt1)

    tau_hat = mu1_hat - mu0_hat

    return tau_hat


def estimate_outcome(xx, tt, e_x_, m_x_, n_runs=1):
    m_samples = xx.shape[0]

    y_t_hat = 0

    for i in range(n_runs):

        y_t_hat = tf_eval(y_hat,m_samples,{input_x: xx, input_t: tt, e_x: e_x_, m_x: m_x_})

    y_t_hat /= n_runs

    return y_t_hat


def eval_y_rmse(yy,yy_mdl):

    yy_mdl_mean = np.mean(yy_mdl,axis=1).reshape([-1,1])

    rmse = np.sqrt(np.mean(np.square(yy-yy_mdl_mean)))

    return rmse




# ihdp_ind = 8



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


Y0 = Y0.reshape([-1,])
Y1 = Y1.reshape([-1,])




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




Y0_test = Y0_test.reshape([-1,])
Y1_test = Y1_test.reshape([-1,])






#----------------------------------------------------------------
#
#     MODELS
#
#----------------------------------------------------------------


# Estimating Entropy balancing weights

x0 = X0
x1 = X1

lam = tf.Variable(0.1*np.random.randn(1,X.shape[1]),dtype=DTYPE)

x0_tf = tf.constant(x0,dtype=DTYPE)
x1_tf = tf.constant(x1,dtype=DTYPE)

x0_lam = tf.multiply(x0_tf,lam)
x1_bar = tf.reduce_mean(x1_tf,axis=0,keepdims=True) # 1 x dim
x1_bar_lam = tf.matmul(lam, tf.transpose(x1_bar))
omega = tf.reduce_sum(x0_lam,axis=1)
t1 = tf.log(tf.reduce_sum(tf.exp(omega)))
t2 = tf.reshape(x1_bar_lam,[])
loss = t1-t2

w_eb = tf.nn.softmax(omega)

learning_rate = tf.placeholder(tf.float32)

train_lam = tf.train.AdamOptimizer(learning_rate).minimize(loss)



# Initialization

initializer = tf.global_variables_initializer()
sess.run(initializer)




# Training

lr = 1e-3

max_epoch = 20
updates_per_epoch = 1000

epoch_record = np.zeros([max_epoch,])

for epoch_id in range(max_epoch):

    loss_record = np.zeros([updates_per_epoch,])

    t0 = time()

    for step in range(updates_per_epoch):

        feed_dict = {learning_rate: lr}

        _, loss_val = sess.run([train_lam, loss], feed_dict)

        loss_record[step] = loss_val

    t1 = time()

    print([epoch_id+1,np.mean(loss_record),t1-t0])
    epoch_record[epoch_id] = np.mean(loss_record)

# _ = plt.plot(epoch_record)



w_eb_val = sess.run(w_eb)

# np.sum(w_eb_val)
# w_eb_val.shape
Wx0 = w_eb_val
Wx1 = np.ones([n1,])/n1


n = X.shape[0]
Wx = np.zeros([n,])
Wx[t0_ind] = Wx0
Wx[t1_ind] = Wx1
Wx = Wx/2*n





XT = np.concatenate([X,T[:,1].reshape([-1,1])],axis=1)

n_trees = 50
model_m = RandomForestRegressor(n_estimators=n_trees)
model_m.fit(XT,Y.reshape([-1,]),Wx)







tau_hat = estimate_causal_effect(model_m, X)
pehe_ = eval_pehe(tau_hat, Tau)*y_std

tau_hat_val = estimate_causal_effect(model_m, X_val)
pehe_val = eval_pehe(tau_hat_val, Tau_val)*y_std

tau_hat_test = estimate_causal_effect(model_m, X_test)
pehe_test = eval_pehe(tau_hat_test, Tau_test)*y_std


print(pehe_)
print(pehe_val)
print(pehe_test)

results = [ihdp_ind,pehe_,pehe_val,pehe_test, y_std]
# results




#----------------------------------------------------------------
#
#     OUTPUT
#
#----------------------------------------------------------------





np.save(str(ihdp_ind)+"_results.npy",results)
