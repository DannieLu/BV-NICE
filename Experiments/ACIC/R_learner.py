

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

#     x_dim = 25
    x_dim = 55
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



NORM = True
VAL = True


lr = 1e-3
max_epoch = 10
updates_per_epoch = 1000



# ihdp_ind = 8
# folder_ind = 1  # 1-77
# file_ind = 1  # 0-99



acic_ind = int(sys.argv[1])
folder_ind = (acic_ind//20)+1   # 1-77
file_ind = np.mod(acic_ind,20)    # 0-99





#----------------------------------------------------------------
#
#     FUNCTIONS
#
#----------------------------------------------------------------



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

def mu_learner(x,t):

    input_tensor = tf.concat([x,tf.cast(t,tf.float32)],axis=-1)

    mu = simple_mlp(input_tensor,FLAGS.y_dim,'mu')

    return mu

def eval_pehe(tau_hat,tau):
    return np.sqrt(np.mean(np.square(tau-tau_hat)))


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


def load_cfdata(file_dir):
    df = pd.read_csv(file_dir)
    z = df['z'].values
    y0 = df['y0'].values
    y1 = df['y1'].values
    y = y0*(1-z) + y1*z
    return [z,y,df['mu0'].values,df['mu1'].values]





def estimate_causal_effect(xx, e_x_, m_x_, n_runs=1):

    m_samples = xx.shape[0]

    t0 = np.zeros([m_samples,FLAGS.t_dim]); t0[:,0] = 1;
    t1 = np.zeros([m_samples,FLAGS.t_dim]); t1[:,1] = 1;
    mu0_hat = estimate_outcome(xx, t0, e_x_, m_x_, n_runs)
    mu1_hat = estimate_outcome(xx, t1, e_x_, m_x_, n_runs)

    tau_hat = mu1_hat - mu0_hat

    return tau_hat

def estimate_outcome(xx, tt, e_x_, m_x_, n_runs=1):
    m_samples = xx.shape[0]

    y_t_hat = 0

    for i in range(n_runs):

        y_t_hat = tf_eval(y_hat,m_samples,{input_x: xx, input_t: tt, e_x: e_x_, m_x: m_x_})

    y_t_hat /= n_runs

    return y_t_hat

def check_results(x_x,t_x,y_x,e_x_,m_x_,tau_x,msg=''):

    tau_hat = estimate_causal_effect(x_x,e_x_,m_x_).reshape([-1,])
    pehe_mkl = eval_pehe(tau_hat, tau_x)*y_std
    corr_ = np.corrcoef(tau_hat.reshape([-1,]),tau_x.reshape([-1,]))[0,1]
    err_ = np.mean(tau_hat)-4

    Y_mdl = estimate_outcome(x_x,t_x,e_x_,m_x_)
    rmse_ = eval_y_rmse(y_x, Y_mdl)


    print('%sPEHE=%.2f, CORR=%.2f, ERR=%.2f, RMSE=%.2f' % (msg, pehe_mkl, corr_, err_, rmse_) )

    return [pehe_mkl, corr_, err_, rmse_]


def eval_y_rmse(yy,yy_mdl):

    yy_mdl_mean = np.mean(yy_mdl,axis=1).reshape([-1,1])

    rmse = np.sqrt(np.mean(np.square(yy-yy_mdl_mean)))

    return rmse







# folder_ind = 1  # 1-77
# file_ind = 1  # 0-99


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


input_x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.x_dim])
input_t = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.t_dim])
input_y = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.y_dim])

# m_x = linear_mdl(input_x, 1, 'm') # expected outcome
# logit_t_x = linear_mdl(input_x, 2, 'ps') # propensity score
# e_x = tf.reshape(tf.nn.softmax(logit_t_x)[:,1],[FLAGS.batch_size,1])

m_x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.y_dim])
e_x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 1])

tau_x = linear_mdl(input_x, 1, 'tau') # causal effect

input_t_bin = tf.reshape(tf.cast(input_t[:,1],dtype=tf.float32),[-1,1])
r_res = (input_y - m_x) - (tf.cast(input_t_bin, dtype=tf.float32) - e_x) * tau_x

# input_t_bin = tf.reshape(tf.cast(input_t[:,1],dtype=tf.float32),[-1,1])
y_hat = m_x + (tf.cast(input_t_bin, dtype=tf.float32) - e_x) * tau_x

loss_r = tf.reduce_mean(tf.square(r_res))




tau_vars = [v for v in tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.startswith('tau')]


learning_rate = tf.placeholder(tf.float32)

train_tau = tf.train.AdamOptimizer(learning_rate).minimize(loss_r, var_list=tau_vars)


# training


from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor


n_trees = 50
model_e = RandomForestClassifier(n_estimators=n_trees)
model_e.fit(X,T[:,1])

model_m = RandomForestRegressor(n_estimators=n_trees)
model_m.fit(X,Y.reshape([-1,]))

# model.predict_proba(X)[:5]
Ex = model_e.predict_proba(X).reshape([-1,1])
Mx = model_m.predict(X).reshape([-1,1])

Ex_val = model_e.predict_proba(X_val).reshape([-1,1])
Mx_val = model_m.predict(X_val).reshape([-1,1])

Ex_test = model_e.predict_proba(X_test).reshape([-1,1])
Mx_test = model_m.predict(X_test).reshape([-1,1])


# Initialization

initializer = tf.global_variables_initializer()
sess.run(initializer)



# Training




results = np.zeros([max_epoch, 3, 4 ])
epoch_record = np.zeros([max_epoch,])

for epoch_id in range(max_epoch):

    loss_record = np.zeros([updates_per_epoch,])

    t0 = time()

    for step in range(updates_per_epoch):

        # n_samples
        ind = np.random.choice(n_samples,FLAGS.batch_size)
        # ind
        feed_dict = {learning_rate:lr}
        feed_dict[input_x] = X[ind]
        feed_dict[input_y] = Y[ind]
        feed_dict[input_t] = T[ind]
        feed_dict[m_x] = Mx[ind]
        feed_dict[e_x] = Ex[ind]

        _,loss_tau_val = sess.run([train_tau, loss_r], feed_dict)
        loss_record[step] = loss_tau_val


    t1 = time()

    print([epoch_id+1,np.mean(loss_record),t1-t0])
    epoch_record[epoch_id] = np.mean(loss_record)

    res1 = check_results(X,T,Y,Ex,Mx,Tau,msg='training-sample:\t')

    # Val
    res2 = check_results(X_val,T_val,Y_val,Ex_val,Mx_val,Tau_val,msg='validation-sample:\t')

    # Out-of-sample
    res3 = check_results(X_test,T_test,Y_test,Ex_test,Mx_test,Tau_test,msg='test-sample:\t')

    results[epoch_id,0,:] = res1
    results[epoch_id,1,:] = res2
    results[epoch_id,2,:] = res3

# _ = plt.plot(epoch_record)







#----------------------------------------------------------------
#
#     OUTPUT
#
#----------------------------------------------------------------



np.save(str(folder_ind)+'_'+str(file_ind)+"_results.npy",results)
