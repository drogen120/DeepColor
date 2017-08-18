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
        net = slim.repeat(inputs, 1, slim.conv2d, 8, [1, 1], scope='conv1')
        end_points['block1'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 1, slim.conv2d, 12, [1, 1], scope='conv2')
        end_points['block2'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 1, slim.conv2d, 16, [1, 1], scope='conv3')
        end_points['block3'] = net
        # net = slim.max_pool2d(net, [2, 2], scope='pool3')
        colormask = slim.conv2d(net, 3, [1, 1], activation_fn=None, scope='conv4')
        output = inputs * colormask
        # ch_r, ch_g, ch_b, ch_bias = tf.split(colormask, 4, axis = 3)
        # output_r = tf.matmul(inputs, ch_r, transpose_a=False, transpose_b=True)
        # output_r = tf.squeeze(output_r)
        # output_g = tf.matmul(inputs, ch_g, transpose_a=False, transpose_b=True)
        # output_g = tf.squeeze(output_g)
        # output_b = tf.matmul(inputs, ch_b, transpose_a=False, transpose_b=True)
        # output_b = tf.squeeze(output_b)
        # output = tf.stack([output_r, output_g, output_b], axis=3) + ch_bias

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
