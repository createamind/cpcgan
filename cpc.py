import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

from module import Model

class CPC(Model):
    """ Interface """
    def __init__(self, name, args, sess, reuse=False, build_graph=True, log_tensorboard=True):
        self.code_size = args[name]['code_size']
        self.hist_terms = 4
        self.future_terms = 4
        self.image_shape = [64, 64, 3]
        super(CPC, self).__init__(name, args, sess=sess, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)
        print("CPC initialized")

    def encode(self, x):
        z = self._encode(self.x)
        return self.sess.run(z, feed_dict={self.x: x})

    def optimize(self, feed_dict):
        if self.log_tensorboard:
            training_step, _, summary = self.sess.run([self.global_step, self.opt_op, self.merged_op], feed_dict=feed_dict)
            self.writer.add_summary(summary, training_step)
        else:
            self.sess.run(self.opt_op, feed_dict=feed_dict)

    """ Implementation """
    def _build_graph(self):
        self._setup_placeholder()

        x_history_flat = tf.reshape(self.x_history, [-1, *self.image_shape])
        x_future_flat = tf.reshape(self.x_future, [-1, *self.image_shape])
        z_history_flat = self._encode(x_history_flat)
        z_future_flat = self._encode(x_future_flat, reuse=True)

        z_history = tf.reshape(z_history_flat, [-1, self.hist_terms, self.code_size])
        z_future = tf.reshape(z_future_flat, [-1, self.future_terms, self.code_size])
        
        context = self._autoregressive(z_history)

        self.logits, self.loss = self._loss(context, z_future)
        self.opt_op = self._optimize(self.loss)

    def _setup_placeholder(self):
        with tf.name_scope('placeholder'):
            self.x_history = tf.placeholder(tf.float32, [None, self.hist_terms, *self.image_shape], name='x_history')
            self.x_future = tf.placeholder(tf.float32, [None, self.future_terms, *self.image_shape], name='x_future')
            self.y = tf.placeholder(tf.bool, [None, 1], name='labels')
            self.is_training = tf.placeholder(tf.bool, (None), name='is_training')
            self.x = tf.placeholder(tf.float32, [None, *self.image_shape], 'x')

    def _encode(self, x, reuse=None):
        with tf.variable_scope('encoder', reuse=self.reuse if reuse is None else reuse):
            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=tf.layers.batch_normalization, activation=lambda x: tf.nn.leaky_relu(x, 0.3))
            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=tf.layers.batch_normalization, activation=lambda x: tf.nn.leaky_relu(x, 0.3))
            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=tf.layers.batch_normalization, activation=lambda x: tf.nn.leaky_relu(x, 0.3))
            x = self._conv_norm_activation(x, 64, 3, 2, padding='valid', normalization=tf.layers.batch_normalization, activation=lambda x: tf.nn.leaky_relu(x, 0.3))

            x = tf.reshape(x, [-1, 64 * 3 * 3])
            x = self._dense_norm_activation(x, 256, normalization=tf.layers.batch_normalization, activation=lambda x: tf.nn.leaky_relu(x, 0.3))
            x = self._dense(x, self.code_size)

        return x

    def _autoregressive(self, x):
        with tf.variable_scope('autoregressive', reuse=self.reuse):
            x = tk.layers.GRU(256, return_sequences=False, name='ar_context')(x)

        return x

    def _loss(self, context, z_future):
        predictions = []
        with tf.variable_scope('prediction', reuse=self.reuse):
            for _ in range(self.future_terms):
                x = self._dense(context, self.code_size)
                predictions.append(x)
            predictions = tf.stack(predictions, axis=1)

        with tf.name_scope('loss'):
            logits = tf.reduce_mean(predictions * z_future, axis=-1)
            logits = tf.reduce_mean(logits, axis=-1, keepdims=True)

            loss = tf.losses.sigmoid_cross_entropy(self.y, logits)

            if self.log_tensorboard:
                tf.summary.scalar('loss_', loss)

        return logits, loss

