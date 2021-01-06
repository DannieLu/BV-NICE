"""
Copyright (C) 2018  Patrick Schwab, ETH Zurich
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np
# from perfect_match.models.baselines.ganite_package.ganite_builder import GANITEBuilder

# from .baseline import Baseline
# from perfect_match.models.baselines.ganite_package.ganite_model import GANITEModel


from .util import get_nonlinearity_by_name, build_mlp


import pandas as pd
from functools import partial

class Baseline(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def to_data_frame(x):
        return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=np.arange(x.shape[1]))

    def _build(self, **kwargs):
        return None

    def build(self, **kwargs):
        self.model = self._build(**kwargs)

    def preprocess(self, x):
        return x

    def postprocess(self, y):
        return y

    def load(self, path):
        pass

    def save(self, path):
        pass

    def predict_for_model(self, model, x):
        if hasattr(self.model, "predict_proba"):
            return self.postprocess(model.predict_proba(self.preprocess(x)))
        else:
            return self.postprocess(model.predict(self.preprocess(x)))

    def predict(self, x):
        return self.predict_for_model(self.model, x)

    def fit_generator_for_model(self, model, train_generator, train_steps, val_generator, val_steps, num_epochs):
        x, y = self.collect_generator(train_generator, train_steps)
        model.fit(x, y)

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        self.fit_generator_for_model(self.model, train_generator, train_steps, val_generator, val_steps, num_epochs)

    def collect_generator(self, generator, generator_steps):
        all_outputs = []
        for _ in range(generator_steps):
            generator_output = next(generator)
            x, y = generator_output[0], generator_output[1]
            all_outputs.append((self.preprocess(x), y))
        return map(partial(np.concatenate, axis=0), zip(*all_outputs))


class GANITE(Baseline):
    def __init__(self):
        super(GANITE, self).__init__()
        self.callbacks = []

    def load(self, path):
        self.model.load(path)

    def _build(self, **kwargs):
        self.best_model_path = kwargs["best_model_path"]
        self.learning_rate = kwargs["learning_rate"]
        self.dropout = kwargs["dropout"]
        self.l2_weight = kwargs["l2_weight"]
        self.num_units = kwargs["num_units"]
        self.num_layers = kwargs["num_layers"]
        self.num_treatments = kwargs["num_treatments"]
        self.imbalance_loss_weight = kwargs["imbalance_loss_weight"]
        self.early_stopping_patience = kwargs["early_stopping_patience"]
        self.early_stopping_on_pehe = kwargs["early_stopping_on_pehe"]
        self.input_dim = kwargs["input_dim"]
        self.output_dim = kwargs["output_dim"]
        self.ganite_weight_alpha = kwargs["ganite_weight_alpha"]
        self.ganite_weight_beta = kwargs["ganite_weight_beta"]
        return GANITEModel(self.input_dim,
                           self.output_dim,
                           num_units=self.num_units,
                           dropout=self.dropout,
                           l2_weight=self.l2_weight,
                           learning_rate=self.learning_rate,
                           num_layers=self.num_layers,
                           num_treatments=self.num_treatments,
                           with_bn=False,
                           nonlinearity="elu",
                           alpha=self.ganite_weight_alpha,
                           beta=self.ganite_weight_beta)

    def fit_generator(self, train_generator, train_steps, val_generator, val_steps, num_epochs, batch_size):
        # num_epochs = int(np.ceil(3000 / batch_size))
        self.model.train(train_generator,
                         train_steps,
                         num_epochs=num_epochs,
                         learning_rate=self.learning_rate,
                         val_generator=val_generator,
                         val_steps=val_steps,
                         dropout=self.dropout,
                         l2_weight=self.l2_weight,
                         imbalance_loss_weight=self.imbalance_loss_weight,
                         checkpoint_path=self.best_model_path,
                         early_stopping_patience=self.early_stopping_patience,
                         early_stopping_on_pehe=self.early_stopping_on_pehe)


class GANITEBuilder(object):
    @staticmethod
    def build(input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
              num_treatments=2, with_bn=False, nonlinearity="elu", initializer=tf.variance_scaling_initializer(),
              alpha=1.0, beta=1.0):
        x = tf.placeholder("float", shape=[None, input_dim], name='x')
        t = tf.placeholder("float", shape=[None, 1], name='t')
        y_f = tf.placeholder("float", shape=[None, 1], name='y_f')
        y_full = tf.placeholder("float", shape=[None, num_treatments], name='y_full')

        y_pred_cf, propensity_scores, z_g = GANITEBuilder.build_counterfactual_block(input_dim, x, t, y_f,
                                                                                     num_units, dropout, l2_weight,
                                                                                     learning_rate, num_layers,
                                                                                     num_treatments, with_bn,
                                                                                     nonlinearity, initializer)

        y_pred_ite, d_ite_pred, d_ite_true, z_i = GANITEBuilder.build_ite_block(input_dim, x, t, y_f, y_full,
                                                                                num_units, dropout, l2_weight,
                                                                                learning_rate, num_layers,
                                                                                num_treatments, with_bn,
                                                                                nonlinearity, initializer)

        # Build losses and optimizers.
        t_one_hot = tf.one_hot(tf.cast(t, "int32"), num_treatments)

        propensity_loss_cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=propensity_scores,
                                                                                    labels=t_one_hot))

        batch_size = tf.shape(y_pred_cf)[0]
        indices = tf.stack([tf.range(batch_size), tf.cast(t, "int32")[:, 0]], axis=-1)
        y_f_pred = tf.gather_nd(y_pred_cf, indices)

        y_f_i = y_f  # tf.Print(y_f, [y_f[:, 0]], message="y_f=", summarize=8)
        y_f_pred_i = y_f_pred  # tf.Print(y_f_pred, [y_f_pred], message="y_f_pred=", summarize=8)

        supervised_loss_cf = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_f_i[:, 0], y_f_pred_i)))

        cf_discriminator_loss = propensity_loss_cf
        cf_generator_loss = -propensity_loss_cf + alpha * supervised_loss_cf

        # D_ITE goal: 0 when True, 1 when Pred
        ite_loss = tf.reduce_mean(tf.log(d_ite_true)) + tf.reduce_mean(tf.log(1 - d_ite_pred))

        y_full_i = y_full  # tf.Print(y_full, [y_full], message="y_full=", summarize=8)
        y_pred_ite_i = y_pred_ite  # tf.Print(y_pred_ite, [y_pred_ite], message="y_pred_ite=", summarize=8)
        supervised_loss_ite = tf.sqrt(tf.reduce_mean(tf.squared_difference(y_full_i, y_pred_ite_i)))

        ite_discriminator_loss = -ite_loss
        ite_generator_loss = ite_loss + beta * supervised_loss_ite
        return cf_generator_loss, cf_discriminator_loss, ite_generator_loss, ite_discriminator_loss, \
               x, t, y_f, y_full, y_pred_cf, y_pred_ite, z_g, z_i

    @staticmethod
    def build_tarnet(mlp_input, t, input_dim, num_layers, num_units, dropout, num_treatments, nonlinearity):
        initializer = tf.variance_scaling_initializer()
        x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]

        all_indices, outputs = [], []
        for i in range(num_treatments):
            indices = tf.reshape(tf.to_int32(tf.where(tf.equal(tf.reshape(t, (-1,)), i))), (-1,))
            current_last_layer_h = tf.gather(x, indices)

            last_layer = build_mlp(current_last_layer_h, num_layers, num_units, dropout, nonlinearity)

            output = tf.layers.dense(last_layer, units=num_treatments, use_bias=True,
                                     bias_initializer=initializer)

            all_indices.append(indices)
            outputs.append(output)
        return tf.concat(outputs, axis=-1), all_indices

    @staticmethod
    def build_counterfactual_block(input_dim, x, t, y_f, num_units=128, dropout=0.0, l2_weight=0.0,
                                   learning_rate=0.0001, num_layers=2,
                                   num_treatments=2, with_bn=False, nonlinearity="elu",
                                   initializer=tf.variance_scaling_initializer()):

        y_pred, z_g = GANITEBuilder.build_counterfactual_generator(input_dim, x, t, y_f, num_units,
                                                                   dropout, l2_weight, learning_rate,
                                                                   num_layers, num_treatments, with_bn,
                                                                   nonlinearity,
                                                                   initializer)

        propensity_scores = GANITEBuilder.build_counterfactual_discriminator(input_dim, x, t, y_pred, num_units,
                                                                             dropout, l2_weight, learning_rate,
                                                                             num_layers, num_treatments, with_bn,
                                                                             nonlinearity,
                                                                             initializer)
        return y_pred, propensity_scores, z_g

    @staticmethod
    def build_counterfactual_generator(input_dim, x, t, y_f, num_units=128, dropout=0.0, l2_weight=0.0,
                                       learning_rate=0.0001, num_layers=2,
                                       num_treatments=2, with_bn=False, nonlinearity="elu",
                                       initializer=tf.variance_scaling_initializer()):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("g_cf",
                               initializer=initializer):
            z_g = tf.placeholder("float", shape=[None, num_treatments-1], name='z_g')

            mlp_input = tf.concat([x, y_f, t, z_g], axis=-1)
            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            y = tf.layers.dense(x, units=num_treatments, use_bias=True,
                                bias_initializer=initializer)
            return y, z_g

    @staticmethod
    def build_counterfactual_discriminator(input_dim, x, t, y_pred, num_units=128, dropout=0.0, l2_weight=0.0,
                                           learning_rate=0.0001, num_layers=2,
                                           num_treatments=2, with_bn=False, nonlinearity="elu",
                                           initializer=tf.variance_scaling_initializer(),
                                           reuse=False):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("d_cf",
                               reuse=reuse,
                               initializer=initializer):
            mlp_input = tf.concat([x, y_pred], axis=-1)
            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            propensity_scores = tf.layers.dense(x, units=num_treatments, use_bias=True,
                                                bias_initializer=initializer)
            return propensity_scores


    @staticmethod
    def build_ite_block(input_dim, x, t, y_f, y_full, num_units=128, dropout=0.0, l2_weight=0.0,
                        learning_rate=0.0001, num_layers=2,
                        num_treatments=2, with_bn=False, nonlinearity="elu",
                        initializer=tf.variance_scaling_initializer()):
        y_pred_ite, z_i = GANITEBuilder.build_ite_generator(input_dim, x, t, y_f, num_units,
                                                        dropout, l2_weight, learning_rate,
                                                        num_layers, num_treatments, with_bn,
                                                        nonlinearity, initializer)

        d_ite_pred = GANITEBuilder.build_ite_discriminator(input_dim, x, t, y_pred_ite, num_units,
                                                           dropout, l2_weight, learning_rate,
                                                           num_layers, num_treatments, with_bn,
                                                           nonlinearity, initializer, reuse=False)

        d_ite_true = GANITEBuilder.build_ite_discriminator(input_dim, x, t, y_full, num_units,
                                                           dropout, l2_weight, learning_rate,
                                                           num_layers, num_treatments, with_bn,
                                                           nonlinearity, initializer, reuse=True)

        return y_pred_ite, d_ite_pred, d_ite_true, z_i

    @staticmethod
    def build_ite_generator(input_dim, x, t, y_f, num_units=128, dropout=0.0, l2_weight=0.0,
                            learning_rate=0.0001, num_layers=2,
                            num_treatments=2, with_bn=False, nonlinearity="elu",
                            initializer=tf.variance_scaling_initializer()):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("g_ite",
                               initializer=initializer):
            z_i = tf.placeholder("float", shape=[None, num_treatments], name='z_i')
            mlp_input = tf.concat([x, z_i], axis=-1)
            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            y_pred = tf.layers.dense(x, units=num_treatments, use_bias=True,
                                     bias_initializer=initializer)
            return y_pred, z_i

    @staticmethod
    def build_ite_discriminator(input_dim, x, t, y_pred, num_units=128, dropout=0.0, l2_weight=0.0,
                                learning_rate=0.0001, num_layers=2,
                                num_treatments=2, with_bn=False, nonlinearity="elu",
                                initializer=tf.variance_scaling_initializer(),
                                reuse=False):
        nonlinearity = get_nonlinearity_by_name(nonlinearity)
        with tf.variable_scope("d_ite",
                               reuse=reuse,
                               initializer=initializer):
            mlp_input = tf.concat([x, y_pred], axis=-1)
            x = build_mlp(mlp_input, num_layers, num_units, dropout, nonlinearity)[0]
            y = tf.layers.dense(x, units=1, use_bias=True,
                                bias_initializer=initializer, activation=tf.nn.sigmoid)
            return y


class GANITEModel(object):
    def __init__(self, input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
                 num_treatments=2, with_bn=False, nonlinearity="elu", initializer=tf.variance_scaling_initializer(),
                 alpha=1.0, beta=1.0):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.num_treatments = num_treatments

        self.cf_generator_loss, self.cf_discriminator_loss, \
        self.ite_generator_loss, self.ite_discriminator_loss, \
        self.x, self.t, self.y_f, self.y_full, self.y_pred_cf, self.y_pred_ite, self.z_g, self.z_i = \
            GANITEBuilder.build(input_dim, output_dim,
                                num_units=num_units,
                                dropout=dropout,
                                l2_weight=l2_weight,
                                learning_rate=learning_rate,
                                num_layers=num_layers,
                                num_treatments=num_treatments,
                                with_bn=with_bn,
                                nonlinearity=nonlinearity,
                                initializer=initializer,
                                alpha=alpha,
                                beta=beta)

    @staticmethod
    def get_scoped_variables(scope_name):
        t_vars = tf.trainable_variables()
        vars = [var for var in t_vars if scope_name in var.name]
        return vars

    @staticmethod
    def get_cf_generator_vairables():
        return GANITEModel.get_scoped_variables("g_cf")

    @staticmethod
    def get_cf_discriminator_vairables():
        return GANITEModel.get_scoped_variables("d_cf")

    @staticmethod
    def get_ite_generator_vairables():
        return GANITEModel.get_scoped_variables("g_ite")

    @staticmethod
    def get_ite_discriminator_vairables():
        return GANITEModel.get_scoped_variables("d_ite")

    def load(self, path):
        saver = tf.train.Saver()
        # saver.restore(self.sess, path)

    def train(self, train_generator, train_steps, val_generator, val_steps, num_epochs,
              learning_rate, learning_rate_decay=0.97, iterations_per_decay=100,
              dropout=0.0, imbalance_loss_weight=0.0, l2_weight=0.0, checkpoint_path="",
              early_stopping_patience=12, early_stopping_on_pehe=False):

        saver = tf.train.Saver(max_to_keep=0)

        global_step_1 = tf.Variable(0, trainable=False, dtype="int64")
        global_step_2 = tf.Variable(0, trainable=False, dtype="int64")
        global_step_3 = tf.Variable(0, trainable=False, dtype="int64")
        global_step_4 = tf.Variable(0, trainable=False, dtype="int64")

        opt = tf.train.AdamOptimizer(learning_rate)
        train_step_g_cf = opt.minimize(self.cf_generator_loss, global_step=global_step_1,
                                       var_list=GANITEModel.get_cf_generator_vairables())
        train_step_d_cf = opt.minimize(self.cf_discriminator_loss, global_step=global_step_2,
                                       var_list=GANITEModel.get_cf_discriminator_vairables())
        train_step_g_ite = opt.minimize(self.ite_generator_loss, global_step=global_step_3,
                                        var_list=GANITEModel.get_ite_generator_vairables())
        train_step_d_ite = opt.minimize(self.ite_discriminator_loss, global_step=global_step_4,
                                        var_list=GANITEModel.get_ite_discriminator_vairables())

        self.sess.run(tf.global_variables_initializer())

        best_val_loss, num_epochs_without_improvement = np.finfo(float).max, 0
        for epoch_idx in range(num_epochs):
            for step_idx in range(train_steps):
                train_losses_g = self.run_generator(train_generator, 1, self.cf_generator_loss, train_step_g_cf)
                train_losses_d = self.run_generator(train_generator, 1, self.cf_discriminator_loss, train_step_d_cf)

            val_losses_g = self.run_generator(val_generator, val_steps, self.cf_generator_loss)
            val_losses_d = self.run_generator(val_generator, val_steps, self.cf_discriminator_loss)

            current_val_loss = val_losses_g[0]
            do_save = current_val_loss < best_val_loss
            if do_save:
                num_epochs_without_improvement = 0
                best_val_loss = current_val_loss
                # saver.save(self.sess, checkpoint_path)
            else:
                num_epochs_without_improvement += 1

            self.print_losses(epoch_idx, num_epochs,
                              [train_losses_g[0], train_losses_d[0]],
                              [val_losses_g[0], val_losses_d[0]],
                              do_save)

            if num_epochs_without_improvement >= early_stopping_patience:
                break

        best_val_loss, num_epochs_without_improvement = np.finfo(float).max, 0
        for epoch_idx in range(num_epochs):
            for step_idx in range(train_steps):
                train_losses_g = self.run_generator(train_generator, 1, self.ite_generator_loss, train_step_g_ite,
                                                    include_y_full=True)
                train_losses_d = self.run_generator(train_generator, 1, self.ite_discriminator_loss, train_step_d_ite,
                                                    include_y_full=True)
            val_losses_g = self.run_generator(val_generator, val_steps, self.ite_generator_loss,
                                              include_y_full=True)
            val_losses_d = self.run_generator(val_generator, val_steps, self.ite_discriminator_loss,
                                              include_y_full=True)

            current_val_loss = val_losses_g[0]
            do_save = current_val_loss < best_val_loss
            if do_save:
                num_epochs_without_improvement = 0
                best_val_loss = current_val_loss
                # saver.save(self.sess, checkpoint_path)
            else:
                num_epochs_without_improvement += 1

            self.print_losses(epoch_idx, num_epochs,
                              [train_losses_g[0], train_losses_d[0]],
                              [val_losses_g[0], val_losses_d[0]],
                              do_save)

            if num_epochs_without_improvement >= early_stopping_patience:
                break

    def print_losses(self, epoch_idx, num_epochs, train_losses, val_losses, did_save=False):
        print("Epoch [{:04d}/{:04d}] {:} TRAIN: G={:.3f} D={:.3f} VAL: G={:.3f} D={:.3f}"
              .format(
                  epoch_idx, num_epochs,
                  "xx" if did_save else "::",
                  train_losses[0], train_losses[1],
                  val_losses[0], val_losses[1]
              ),
              file=sys.stderr)

    def run_generator(self, generator, steps, loss, train_step=None, include_y_full=False):
        losses = []
        for iter_idx in range(steps):
            (x_batch, t_batch), y_batch = next(generator)
            t_batch = np.expand_dims(t_batch, axis=-1)
            y_batch = np.expand_dims(y_batch, axis=-1)

            batch_size = len(x_batch)
            feed_dict = {
                self.x: x_batch,
                self.t: t_batch,
                self.y_f: y_batch,
                self.z_g: np.random.uniform(size=(batch_size, self.num_treatments-1)),
                self.z_i: np.random.uniform(size=(batch_size, self.num_treatments))
            }
            if include_y_full:
                y_pred = self._predict_g_cf([x_batch, t_batch], y_batch)
                # print(len(y_pred))
                y_pred[np.arange(len(y_pred)), np.array(t_batch,dtype=np.int32).reshape([-1,])] = y_batch.reshape([-1,])
                # y_pred[np.arange(len(y_pred)), t_batch.reshape([-1,])] = y_batch
                feed_dict[self.y_full] = y_pred

            if train_step is not None:
                self.sess.run(train_step, feed_dict=feed_dict)

            losses.append(self.sess.run([loss],
                                        feed_dict=feed_dict))
        return np.mean(losses, axis=0)

    def _predict_g_cf(self, x, y_f):
        batch_size = len(x[0])
        y_pred = self.sess.run(self.y_pred_cf, feed_dict={
            self.x: x[0],
            self.t: x[1],
            self.y_f: y_f,
            self.z_g: np.random.uniform(size=(batch_size, self.num_treatments-1))
        })
        return y_pred

    def predict(self, x):
        batch_size = len(x[0])
        y_pred = self.sess.run(self.y_pred_ite, feed_dict={
            self.x: x[0],
            self.z_i: np.random.uniform(size=(batch_size, self.num_treatments))
        })
        # y_pred = np.array(map(lambda inner, idx: inner[idx], y_pred, x[1]))
        return y_pred