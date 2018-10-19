import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utils import tf_utils
from module import Module
from cpc import CPC
from wgangp import WGANGP

class CPCGAN(Module):
    """ Interface """
    def __init__(self, name, args, sess=None, reuse=False, build_graph=True, 
                 log_tensorboard=False, save=True, loss_type='supervised', 
                 tensorboard_root_dir='/tmp/cpcgan/tensorboard_logs'):
        self.image_shape = args['image_shape']
        self.batch_size = args['batch_size']
        self.gan_coeff = args['gan_coeff']
        self.cpc_coeff = args['cpc_coeff']

        self.sess = sess if sess is not None else tf.get_default_session()
        self._save = save

        super().__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

        if self._log_tensorboard:
            self.graph_summary, self.writer = self._setup_tensorboard_summary(tensorboard_root_dir)

    def optimize_cpc(self, history, future, training, label):
        feed_dict = self._construct_feed_dict(history, future, training, label)
        if self._log_tensorboard:
            train_steps, logits, loss, _, summary = self.sess.run([self.cpc.train_steps, self.cpc.logits, self.cpc.loss, 
                                                self.cpc.opt_op, self.graph_summary], feed_dict=feed_dict)
            if self._time_to_save(train_steps):
                self.writer.add_summary(summary, train_steps)
        else:
            logits, loss = self.sess.run([self.cpc.logits, self.cpc.loss], feed_dict=feed_dict)

        return logits, loss

    def optimize_gan(self, history, future, training, label):
        feed_dict = self._construct_feed_dict(history, future, training, label)
        if self._log_tensorboard:
            train_steps, generator_loss, critic_loss, _, summary = self.sess.run([self.gan_train_steps, self.generator_loss, self.critic_loss, 
                                                    self.gan_opt_op, self.graph_summary], feed_dict=feed_dict)
            if self._time_to_save(train_steps):
                self.writer.add_summary(summary, train_steps)
        else:
            generator_loss, critic_loss, _ = self.sess.run(self.generator_loss, self.critic_loss, self.gan_opt_op, feed_dict=feed_dict)

        return generator_loss, critic_loss

    """ restore & save """
    def restore(self):
        self.restore_cpc()
        self.restore_gan()

    def restore_cpc(self):
        self.cpc.restore()
    
    def restore_gan(self):
        self.gan.restore()

    def save(self):
        self.save_cpc()
        self.save_gan()

    def save_cpc(self):
        self.cpc.save()

    def save_gan(self):
        self.gan.save()

    """ Implementation """
    def _build_graph(self):
        with tf.name_scope('placeholder'):
            self._training = tf.placeholder(tf.bool, [], name='training')

        cpc_args = self._add_model_to_args(self._args['cpc'])

        self.cpc = CPC('cpc', cpc_args, 
                       self.batch_size, self.image_shape,
                       self._args['code_size'], training=self._training,
                       sess=self.sess, reuse=self._reuse, 
                       build_graph=self._build_graph, 
                       log_tensorboard=self._log_tensorboard,
                       save=self._save, scope_prefix=self.name)
        
        self.gans, self.generated_images = self._gans()

        self.gan = self.gans[0]

        self.gancpc = CPC('cpc', self._args['cpc'],
                          self.batch_size, self.image_shape,
                          self._args['code_size'], training=self._training,
                          x_future=self.generated_images,
                          sess=self.sess, reuse=True, 
                          build_graph=self._build_graph,
                          log_tensorboard=False,
                          save=False, scope_prefix=self.name)

        self.generator_loss, self.critic_loss = self._gan_loss(self.gans)

        self.gan_opt_op = self._gan_opt_op(self.gan, self.generator_loss, self.critic_loss)

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
                        training=self._training, sess=self.sess,
                        reuse=reuse, build_graph=self._build_graph, 
                        log_tensorboard=log_tensorboard,
                        save=save, scope_prefix=self.name)

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
            self.gan_train_steps = tf.get_variable('gan_train_steps', shape=(), initializer=tf.constant_initializer([0]), trainable=False)
            step_op = tf.assign(self.gan_train_steps, self.gan_train_steps + 1, name='gan_update_train_step')

        with tf.variable_scope('gan_optimizer', reuse=self._reuse):
            generator_opt_op = gan.generator.optimize_op(generator_loss)
            critic_opt_op = gan.real_critic.optimize_op(critic_loss)
            
            with tf.control_dependencies([step_op]):
                opt_op = tf.group(generator_opt_op, critic_opt_op)

        return opt_op

    def _time_to_save(self, train_steps):
        return train_steps % 10 == 0

    def _setup_tensorboard_summary(self, root_dir):
        graph_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(os.path.join(root_dir, self._args['model_dir'], self._args['model_name']), self.sess.graph)

        return graph_summary, writer

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
