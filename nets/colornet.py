from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tf_extended as tfe

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

ColorNetParams = namedtuple('ColorNetParams', ['img_shape',
                                         ])

def color_net(inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='color_net'):
    """color net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'color_net', [inputs], reuse=reuse):
        # bs = 2

        global_net = slim.repeat(inputs, 1, slim.conv2d, 8, [3, 3], scope='g_conv1')
        global_net = slim.max_pool2d(global_net, [2, 2], scope='g_pool1')
        global_net = slim.repeat(global_net, 1, slim.conv2d, 16, [3, 3], scope='g_conv2')
        global_net = slim.max_pool2d(global_net, [2, 2], scope='g_pool2')
        global_net = slim.repeat(global_net, 1, slim.conv2d, 32, [3, 3], scope='g_conv3')
        global_net = slim.max_pool2d(global_net, [2, 2], scope='g_pool3')
        global_net = slim.repeat(global_net, 1, slim.conv2d, 32, [3, 3], scope='g_conv4')
        global_net = slim.max_pool2d(global_net, [2, 2], scope='g_pool4')
        global_net = slim.repeat(global_net, 1, slim.conv2d, 64, [3, 3], scope='g_conv5')
        global_net = slim.max_pool2d(global_net, [2, 2], scope='g_pool5')
        global_net = slim.repeat(global_net, 1, slim.conv2d, 128, [3, 3], scope='g_conv6')
        global_net = slim.max_pool2d(global_net, [2, 2], scope='g_pool6')
        global_net = slim.flatten(global_net)
        global_net = slim.fully_connected(global_net, 256, scope='g_fc1')
        global_net = slim.fully_connected(global_net, 64, scope='g_fc2')
        output = slim.fully_connected(global_net, 12, activation_fn=None, scope='output')
        # global_net = tf.reshape(global_net, [bs, 1, 1, 64])


        return output, end_points

def colornet_arg_scope(weight_decay=0.0005, data_format='NHWC', is_training = True):
    """Defines the colornet arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.99},
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format) as sc:
            return sc
def colornet_losses(groundtrouth, predicts, scope=None):
    with tf.name_scope(scope, 'colornet_losses'):
        # lshape = tfe.get_shape(groundtrouth[0], 5)
        # num_classes = lshape[-1]
        # batch_size = lshape[0]
        with tf.name_scope('l2_loss'):
            loss = tf.nn.l2_loss(predicts-groundtrouth)
            loss = tf.div(loss, 5000.0, name="l2_value")
            tf.losses.add_loss(loss)

color_net.default_image_size = 512
class ColorNet(object):
    default_params = ColorNetParams(img_shape = (512, 512))

    def __init__(self, params=None):
        if isinstance(params, ColorNetParams):
            self.params = params
        else:
            self.params = ColorNet.default_params

    def net(self, inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            reuse=None,
            scope='color_net'):
        """colornet definition.
        """
        r = color_net(inputs,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse,
                    scope=scope)
        return r


    def arg_scope(self, weight_decay=0.0005, data_format='NHWC', is_training= True):
        """Network arg_scope.
        """
        return colornet_arg_scope(weight_decay, data_format=data_format, is_training=is_training)

    def losses(self, groundtrouth, predicts,
               scope='colornet_losses'):
        """Define the colornet losses.
        """
        return colornet_losses(groundtrouth, predicts, scope=scope)
