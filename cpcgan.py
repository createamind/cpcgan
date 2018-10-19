import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utils import tf_utils
from module import Module, Model
from cpc import CPC
from wgangp import WGANGP

class CPCGAN(Model):
    """ Interface """
    def __init__(self, name, args, sess=None, reuse=False, build_graph=True, 
                 log_tensorboard=False, save=True, loss_type='supervised'):
        self.image_shape = args['image_shape']
        self.batch_size = args['batch_size']
        self.gan_coeff = args['gan_coeff']
        self.cpc_coeff = args['cpc_coeff']

        self._save = save

        super().__init__(name, args, 
                        sess=sess, reuse=reuse, 
                        build_graph=build_graph, 
                        log_tensorboard=log_tensorboard, save=save)

        if self._log_tensorboard:
            # image log
            self.comparison, self.comparison_counter, self.comparison_log_op = self._setup_comparison_log()

        # initialize gan's optimizer annd sequence_counter
        self.sess.run(tf.global_variables_initializer())

        # this saver only stores sequence_counter, cpc and gan have their own saver
        self._saver = self._setup_saver_comparison(save)

    def optimize_cpc(self, history, future, training, label):
        feed_dict = self._construct_feed_dict(history, future, training, label)
        if self._log_tensorboard:
            train_steps, logits, loss, _, summary = self.sess.run([self.cpc.train_steps, self.cpc.logits, 
                                                                    self.cpc.loss, self.cpc.opt_op, 
                                                                    self.graph_summary], feed_dict=feed_dict)
            if self._time_to_save(train_steps):
                self.writer.add_summary(summary, train_steps)
        else:
            logits, loss = self.sess.run([self.cpc.logits, self.cpc.loss], feed_dict=feed_dict)

        return logits, loss

    def optimize_gan(self, history, future, training, label):
        feed_dict = self._construct_feed_dict(history, future, training, label)
        if self._log_tensorboard:
            train_steps, generated_images, generator_loss, critic_loss, _, summary = self.sess.run([self.gan_train_steps, self.generated_images,
                                                                                                    self.generator_loss, self.critic_loss, 
                                                                                                    self.gan_opt_op, self.graph_summary], feed_dict=feed_dict)
            if self._time_to_save(train_steps):
                self._log_comparison(history, future, generated_images)
                self.writer.add_summary(summary, train_steps)

            self.sess.run([self.gan_train_steps, self.graph_summary], feed_dict=feed_dict)
        else:
            generator_loss, critic_loss, _ = self.sess.run([self.generator_loss, self.critic_loss, self.gan_opt_op], feed_dict=feed_dict)

        return generator_loss, critic_loss

    """ restore & save """

    """ Implementation """
    def _build_graph(self):
        with tf.name_scope('placeholder'):
            self._training = tf.placeholder(tf.bool, [], name='training')

        cpc_args = self._add_model_to_args(self._args['cpc'])

        self.cpc = CPC('cpc', cpc_args, 
                       self.batch_size, self.image_shape,
                       self._args['code_size'], training=self._training,
                       reuse=self._reuse, build_graph=self._build_graph, 
                       log_tensorboard=self._log_tensorboard,
                       scope_prefix=self.name)
        
        self.gans, self.generated_images = self._gans()

        self.gan = self.gans[0]

        self.gancpc = CPC('cpc', self._args['cpc'],
                          self.batch_size, self.image_shape,
                          self._args['code_size'], training=self._training,
                          x_future=self.generated_images,
                          reuse=True, build_graph=self._build_graph,
                          log_tensorboard=False,
                          scope_prefix=self.name)

        self.generator_loss, self.critic_loss = self._gan_loss(self.gans)

        self.gan_train_steps, self.gan_opt_op = self._gan_opt_op(self.gan, self.generator_loss, self.critic_loss)
        
    def _gans(self):
        """ This actually returns multiple shallow copies of a single GAN with different input """
        gans = []
        generated_images = []

        gan_args = self._add_model_to_args(self._args['gan'])

        for i in range(self.cpc.future_terms):
            # all other GANs reuse the params of the first
            reuse = self._reuse if i == 0 else True
            log_tensorboard = self._log_tensorboard if i == 0 else False
            save = self._save if i == 0 else False

            gan = WGANGP('gan', gan_args,
                        self.batch_size, self.image_shape,
                        self._args['code_size'],
                        self.cpc.predictions[:, i, ...],
                        self.cpc.x_future[:, i, ...], i,
                        training=self._training,
                        reuse=reuse, build_graph=self._build_graph, 
                        log_tensorboard=log_tensorboard,
                        scope_prefix=self.name)

            generated_image = gan.generator.generated_image
            gans.append(gan)
            generated_images.append(generated_image)
        
        # to be consistent with the shape of cpc.x_future
        generated_images = tf.stack(generated_images, axis=1, name='generated_images')

        return gans, generated_images

    def _gan_loss(self, gans):
        generator_losses = []
        critic_losses = []

        with tf.variable_scope('gan_loss'):
            for i in range(self.cpc.future_terms):
                # consider use MSE instead of cpc_loss on z level or more directly x level
                cpc_loss = tf.losses.sigmoid_cross_entropy(self.gancpc.label, tf.expand_dims(self.gancpc.individual_logits[:, i], axis=1))
                generator_loss = self.gan_coeff * gans[i].generator_loss + self.cpc_coeff * cpc_loss
                critic_loss = self.gan_coeff * gans[i].critic_loss

                generator_losses.append(generator_loss)
                critic_losses.append(critic_loss)

            generator_loss = tf.reduce_mean(generator_losses, name='generator_loss')
            critic_loss = tf.reduce_mean(critic_losses, name='critic_loss')

        if self._log_tensorboard or self._log_tensorboard:
            with tf.name_scope('loss'):
                tf.summary.scalar('generator_loss', generator_loss)
                tf.summary.scalar('critic_loss', critic_loss)

        return generator_loss, critic_loss

    def _gan_opt_op(self, gan, generator_loss, critic_loss):
        with tf.variable_scope('gan_train_steps', reuse=self._reuse):
            gan_train_steps = tf.get_variable('gan_train_steps', shape=(), initializer=tf.constant_initializer([0]), trainable=False)
            step_op = tf.assign(gan_train_steps, gan_train_steps + 1, name='gan_update_train_step')

        with tf.variable_scope('gan_optimizer', reuse=self._reuse):
            generator_opt_op = gan.generator.optimize_op(generator_loss)
            critic_opt_op = gan.real_critic.optimize_op(critic_loss)
            
            with tf.control_dependencies([step_op]):
                opt_op = tf.group(generator_opt_op, critic_opt_op)

        return gan_train_steps, opt_op

    def _time_to_save(self, train_steps):
        return train_steps % 10 == 0

    def _add_model_to_args(self, args):
        model_dict = {
            'model_dir': self._args['model_dir'],
            'model_name': self._args['model_name']
        }
        args.update(model_dict)

        return args

    def _construct_feed_dict(self, history, future, training, label):
        feed_dict = {
            self.cpc.x_history: history,
            self.cpc.x_future: future,
            self.cpc._training: training,
            self.cpc.label: label,
            self.gan._training: training,
            self.gancpc.x_history: history,
            self.gancpc.label: label,
            self.gancpc._training: training,
        }

        return feed_dict

    def _setup_comparison_log(self):
        with tf.variable_scope('comparison', reuse=self._reuse):
            height, width, channels = self.image_shape
            height *= 2
            width *= self.cpc.hist_terms + self.cpc.future_terms

            comparison = tf.placeholder(tf.float32, 
                                        [None, height, width, channels], 
                                        name='comparison')

            comparison_counter = tf.get_variable('comparison_counter', shape=[], initializer=tf.constant_initializer(), trainable=False)
            step_op = tf.assign(comparison_counter, comparison_counter + 1, name='update_comparison_counter')

            comparison_log = tf.summary.image('comparison_', comparison)

            with tf.control_dependencies([step_op]):
                comparison_log_op = tf.summary.merge([comparison_log], name='comparison_log_op')

        return comparison, comparison_counter, comparison_log_op
            
    def _setup_saver_comparison(self, save):
        return tf.train.Saver(tf.global_variables(scope='comparison')) if save else None
        
    def _log_comparison(self, history, future, generated_images):
        images = self._concate_images(history, future, generated_images)

        feed_dict = {
            self.comparison: images
        }
        counter, summary = self.sess.run([self.comparison_counter, self.comparison_log_op], feed_dict=feed_dict)
        self.writer.add_summary(summary, counter)

    def _concate_images(self, history, future, generated_images):
        terms = history.shape[1] + future.shape[1]
        height, width, channels = history.shape[-3:]

        def concate_width(images):
            images = np.transpose(images, [0, 1, 3, 2, 4])
            images = np.reshape(images, (-1, terms * width, height, channels))
            images = np.transpose(images, [0, 2, 1, 3])

            return images
        
        real_seq = np.concatenate((history, future), axis=1)
        real_seq = concate_width(real_seq)
        fake_seq = np.concatenate((history, generated_images), axis=1)
        fake_seq = concate_width(fake_seq)

        output = np.concatenate((real_seq, fake_seq), axis=1)

        return output