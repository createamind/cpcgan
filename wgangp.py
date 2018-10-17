import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

from utils import tf_utils
from module import Model, Module


class WGANGP(Model):
    """ Interface """
    def __init__(self, name, args, encoder, cpc, sess):
        super().__init__(name, args, sess)
        self.image_shape = args['image_shape']
        self.latent_dim = args['code_size']
        self.predict_terms = args['predict_terms']
        self.batch_size = args['batch_size']
        self.critic_coeff = args['critic_coeff']

    """ Implementation """
    def _build_graph(self):
        with tf.variable_scope('inputs', reuse=self._reuse):
            self.z = tf.placeholder(tf.float32, [None, self.latent_dim], name='z')
            self.image = tf.placeholder(tf.float32, [None, *self.image_shape], name='image')
            self._training = tf.placeholder(tf.bool, [], name='training')

        self.generator = Generator('generator', self._generator_args())
        
        # interpolated image
        t = np.random.random(size=(self.batch_size, 1, 1, 1))
        with tf.name_scope('interpolated_image'):
            self.interpolated_image = t * self.generator.generated_image + (1 - t) * self.image

        self.real_critic = Critic('critic', self._critic_args(self.image), reuse=False)
        self.fake_critic = Critic('critic', self._critic_args(self.generator.generated_image), reuse=True)
        self.interpolated_critic = Critic('critic', self._critic_args(self.interpolated_image), reuse=True)

        real_loss, fake_loss, wasserstein_loss = self._wasserstein_loss(self.real_critic.validity, self.fake_critic)
        self.generator_loss = fake_loss
        self.critic_loss = (wasserstein_loss + self.critic_coeff * self._gradient_penalty(self.interpolated_critic.validity, self.interpolated_image))

        self.generator_opt_op = self.optimize(self.generator_loss)
        self.critic_opt_op = self.optimize(self.critic_loss)

    def _generator_args(self):
        args = {
            'image_shape': self.image_shape,
            'latent_dim': self.latent_dim,
            'z': self.z,
        }

        return args

    def _critic_args(self, image):
        args = {
            'image_shape': self.image_shape,
            'image': image,
        }

        return args

    def _wasserstein_loss(self, real, fake):
        real_loss = tf.reduce_mean(real, name='real_loss')
        fake_loss = -tf.reduce_mean(fake, name='fake_loss')
        wasserstein_loss = tf.negative(real_loss + fake_loss, 'wasserstein_loss')
        
        return real_loss, fake_loss, wasserstein_loss

    def _gradient_penalty(self, validity, interpolated_image):
        interpolated_grads = tf.gradients(validity, interpolated_image, name='interpolated_grads')

        grads_l2 = tf.sqrt(tf.reduce_sum(tf.square(interpolated_grads)), name='grads_l2')

        return tf.square(grads_l2 - 1, name='gradient_penalty')


class Generator(Module):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        super.__init__(name, args, reuse)
        self.image_shape = args['image_shape']
        self.latent_dim = args['latent_dim']
        self.z = args['z']

    """ Implementation """
    def _build_graph(self):
        self.generated_image = self._build_generator(self.z)
    
    def _build_generator(self, x):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=.8)

        x = self._dense_norm_activation(x, 128 * 7 * 7, activation=tf.nn.relu)
        x = tf.reshape(x, (7, 7, 128))
        x = tf.image.resize_images(x, (14, 14))
        x = self._conv_norm_activation(x, 128, 4, 
                                       normalization=bn, 
                                       activation=tf.nn.relu)
        x = tf.image.resize_images(x, (28, 28))
        x = self._conv_norm_activation(x, 64, 4,
                                       normalization=bn,
                                       activation=tf.nn.relu)
        x = self._conv_norm_activation(x, self.image_shape[-1], 4, activation=tf.tanh)

        return x


class Critic(Module):
    """ Interface """
    def __init__(self, name, args, reuse=False):
        super.__init__(name, args, reuse)
        self.image_shape = args['image_shape']
        self.image = args['image']

    def _build_graph(self):
        self.validity = self._build_critic(self.image)

    def _build_critic(self, x):
        bn = lambda x: tf.layers.batch_normalization(x, momentum=.8)
        leaky_relu = lambda x: tf.maximum(x, .2 * x)
        dropout = lambda x: tf.layers.dropout(x, .25)

        x = self._conv_norm_activation(x, 16, 3, 2, activation=leaky_relu)
        x = dropout(x)
        x = self._conv_norm_activation(x, 32, 3, 2)
        x = tf.image.pad_to_bounding_box(x, 0, 0, 8, 8)
        x = leaky_relu(bn(x))
        x = dropout(x)
        x = self._conv_norm_activation(x, 64, 3, 2, normalization=bn, activation=leaky_relu)
        x = dropout(x)
        x = self._conv_norm_activation(x, 128, 3, 1, normalization=bn, activation=leaky_relu)
        x = dropout(x)
        x = tf.reshape(x, (128 * 4 * 4))
        x = self._dense_norm_activation(x, 1)

        return x
