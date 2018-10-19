import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from utils import utils, tf_utils
import os
import sys

""" 
Module defines the basic functions to build a tesorflow graph
Model further defines save & restore functionns based onn Module
For example, Actor-Critic should inherit Module and DDPG should inherit Model
since we generally save parameters all together in DDPG
"""

class Module(object):
    """ Interface """
    def __init__(self, name, args, reuse=False, build_graph=True, log_tensorboard=False):
        self.name = name
        self._args = args
        self._reuse = reuse
        self._log_tensorboard = log_tensorboard

        if build_graph:
            self.build_graph()
        
    def build_graph(self):
        with tf.variable_scope(self.name, reuse=self._reuse):
            self._build_graph()

    @property
    def global_variables(self):
        return tf.global_variables(scope=self.name)

    @property
    def trainable_variables(self):
        return tf.trainable_variables(scope=self.name)
        
    @property
    def perturbable_variables(self):
        return [var for var in self.trainable_variables if 'LayerNorm' not in var.name]
        
    @property
    def training(self):
        """ this property should only be used with batch normalization, 
        self._training should be a boolean placeholder """
        return getattr(self, '_training', False)

    @property
    def trainable(self):
        return getattr(self, '_trainable', True)

    @property
    def l2_regularizer(self):
        return tc.layers.l2_regularizer(self._args['weight_decay'] if self.name in self._args and 'weight_decay' in self._args else 0.)
    
    @property
    def l2_loss(self):
        return tf.losses.get_regularization_loss(scope=self.name, name=self.name + 'l2_loss')

    def optimize_op(self, loss, tvars=None):
        with tf.variable_scope(self.name + '_opt', reuse=self._reuse):
            return self._optimize_op(loss, tvars=tvars)
            
    """ Implementation """
    def _build_graph(self):
        raise NotImplementedError
    
    def _optimize_op(self, loss, tvars=None):
        # params for optimizer
        learning_rate = self._args['learning_rate'] if 'learning_rate' in self._args else 1e-3
        beta1 = self._args['beta1'] if 'beta1' in self._args else 0.9
        beta2 = self._args['beta2'] if 'beta2' in self._args else 0.999
        decay_rate = self._args['decay_rate'] if 'decay_rate' in self._args else 1.
        decay_steps = self._args['decay_steps'] if 'decay_steps' in self._args else 1e6

        clip_norm = self._args['clip_norm'] if 'clip_norm' in self._args else 5.

        with tf.variable_scope('optimizer', reuse=self._reuse):
            # setup optimizer
            train_steps = tf.get_variable('train_steps', shape=(), initializer=tf.constant_initializer(), trainable=False)
            if decay_rate != 1.:
                learning_rate = tf.train.exponential_decay(learning_rate, train_steps, decay_steps, decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                tvars = self.trainable_variables if tvars is None else tvars
                grads, tvars = list(zip(*optimizer.compute_gradients(loss, var_list=tvars)))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                opt_op = optimizer.apply_gradients(zip(grads, tvars), global_step=train_steps)

        if self._log_tensorboard:
            if decay_rate != 1:
                tf.summary.scalar('learning_rate_', learning_rate)

            with tf.name_scope('grads'):
                for grad, var in zip(grads, tvars):
                    if grad is not None:
                        tf.summary.histogram(var.name.replace(':0', ''), grad)
            with tf.name_scope('params'):
                for var in self.trainable_variables:
                    tf.summary.histogram(var.name.replace(':0', ''), var)
            
        return train_steps, opt_op
        
    def _dense(self, x, units, kernel_initializer=tf_utils.xavier_initializer(), name=None, reuse=None):
        return tf.layers.dense(x, units, kernel_initializer=kernel_initializer, 
                               kernel_regularizer=self.l2_regularizer, 
                               trainable=self.trainable, 
                               name=name, reuse=reuse)

    def _dense_norm_activation(self, x, units, kernel_initializer=tf_utils.xavier_initializer(),
                               normalization=None, activation=None, name=None, reuse=None):
        x = self._dense(x, units, kernel_initializer=kernel_initializer, name=name, reuse=reuse)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, 
                                     training=self.training, trainable=self.trainable)

        return x

    def _conv(self, x, filters, kernel_size, strides=1, padding='same', 
              kernel_initializer=tf_utils.xavier_initializer(), name=None, reuse=None): 
        return tf.layers.conv2d(x, filters, kernel_size, 
                                strides=strides, padding=padding, 
                                kernel_initializer=kernel_initializer, 
                                kernel_regularizer=self.l2_regularizer, 
                                trainable=self.trainable, name=name, reuse=reuse)

    def _conv_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                              kernel_initializer=tf_utils.xavier_initializer(), normalization=None, 
                              activation=None, name=None, reuse=None):
        x = self._conv(x, filters, kernel_size, 
                       strides=strides, padding=padding, 
                       kernel_initializer=kernel_initializer, 
                       name=name, reuse=reuse)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, 
                                     training=self.training, trainable=self.trainable)

        return x
    
    def _convtrans(self, x, filters, kernel_size, strides=1, padding='same', 
                   kernel_initializer=tf_utils.xavier_initializer(), name=None, reuse=None): 
        return tf.layers.conv2d_transpose(x, filters, kernel_size, 
                                          strides=strides, padding=padding, 
                                          kernel_initializer=kernel_initializer, 
                                          kernel_regularizer=self.l2_regularizer, 
                                          trainable=self.trainable, name=name, reuse=reuse)

    def _convtrans_norm_activation(self, x, filters, kernel_size, strides=1, padding='same', 
                                   kernel_initializer=tf_utils.xavier_initializer(), normalization=None, 
                                   activation=None, name=None, reuse=None):
        x = self._convtrans(x, filters, kernel_size, 
                            strides=strides, padding=padding, 
                            kernel_initializer=kernel_initializer,
                            name=name, reuse=reuse)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, 
                                     training=self.training, trainable=self.trainable)

        return x

    def _noisy(self, x, units, kernel_initializer=tf_utils.xavier_initializer(), 
               name=None, reuse=None, sigma=.4):
        name = name if name is not None else 'noisy'
        
        with tf.variable_scope(name, reuse=reuse):
            y = self._dense(x, units, kernel_initializer=kernel_initializer, reuse=reuse)
            
            with tf.variable_scope('noisy', reuse=reuse):
                # params for the noisy layer
                features = x.shape.as_list()[-1]
                w_shape = [features, units]
                b_shape = [units]
                epsilon_w = tf.truncated_normal(w_shape, stddev=sigma, name='epsilon_w')
                epsilon_b = tf.truncated_normal(b_shape, stddev=sigma, name='epsilon_b')
                noisy_w = tf.get_variable('noisy_w', shape=w_shape, 
                                          initializer=kernel_initializer,
                                          regularizer=self.l2_regularizer, 
                                          trainable=self.trainable)
                noisy_b = tf.get_variable('noisy_b', shape=b_shape, 
                                          initializer=tf.constant_initializer(sigma / np.sqrt(units)), 
                                          trainable=self.trainable)
                
                # output of the noisy layer
                x = tf.matmul(x, noisy_w * epsilon_w) + noisy_b * epsilon_b

            x = x + y

        if self.trainable:
            return x
        else:
            return y

    def _noisy_norm_activation(self, x, units, kernel_initializer=tf_utils.xavier_initializer(),
                               normalization=None, activation=None, 
                               name=None, reuse=None, sigma=.4):
        x = self._noisy(x, units, kernel_initializer=kernel_initializer, 
                        name=name, reuse=reuse, sigma=sigma)
        x = tf_utils.norm_activation(x, normalization=normalization, activation=activation, 
                                     training=self.training, trainable=self.trainable)

        return x


class Model(Module):
    """ Interface """
    def __init__(self, name, args, sess=None, reuse=False, 
                 build_graph=True, log_tensorboard=False, save=True, 
                 model_root_dir='/tmp/imin/saved_models', 
                 tensorboard_root_dir='/tmp/imin/tensorboard_logs'):
        # initialize session and global variables
        self.sess = sess if sess is not None else tf.get_default_session()

        super(Model, self).__init__(name, args, reuse, build_graph, log_tensorboard)
            
        if build_graph:
            self.sess.run(tf.global_variables_initializer())
    
        self._saver = self._setup_saver(save)

        if self._saver:
            self.model_name, self.model_dir, self.model_file = self._setup_model_path(root_dir=model_root_dir)

        if self._log_tensorboard:
            self.graph_summary, self.writer = self._setup_tensorboard_summary(tensorboard_root_dir)
    
    def restore(self):
        """ To restore the most recent model, simply leave filename None
        To restore a specific version of model, set filename to the model stored in saved_models
        """
        try:
            self._saver.restore(self.sess, self.model_file)
        except:
            print('Model {}: No saved model for "{}" is found. \nStart Training from Scratch!'.format(self.model_name, self.name))
        else:
            print("Model {}: Params for {} are restored.".format(self.model_name, self.name))

    def save(self):
        if self._saver:
            self._saver.save(self.sess, self.model_file)

    """ Implementation """
    def _setup_saver(self, save):
        return tf.train.Saver(self.global_variables) if save else None

    def _setup_model_path(self, root_dir):
        model_dir = os.path.join(root_dir, self._args['model_dir'])

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        
        model_name = self._args['model_name']
        model_file = os.path.join(model_dir, model_name)

        return model_name, model_dir, model_file

    def _setup_tensorboard_summary(self, root_dir):
        graph_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(root_dir, self._args['model_dir'], self._args['model_name']), self.sess.graph)

        return graph_summary, writer
