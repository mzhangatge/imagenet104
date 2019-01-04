from __future__ import print_function

try:
    from builtins import range
except ImportError:
    pass
import tensorflow as tf
import numpy as np
from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import batching
import horovod.tensorflow as hvd
import os
import sys
import time
import random
import shutil
import logging
import math
import re
from glob import glob
from operator import itemgetter
from tensorflow.python.util import nest

class LayerBuilder(object):
    def __init__(self, activation=None, data_format='channels_last',
                 training=False, use_batch_norm=False, batch_norm_config=None,
                 conv_initializer=None, adv_bn_init=False):
        self.activation = activation
        self.data_format = data_format
        self.training = training
        self.use_batch_norm = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self.conv_initializer = conv_initializer
        self.adv_bn_init = adv_bn_init
        if self.batch_norm_config is None:
            self.batch_norm_config = {
                'decay': 0.9,
                'epsilon': 1e-4,
                'scale': True,
                'zero_debias_moving_mean': False,
            }

    def _conv2d(self, inputs, activation, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=not self.use_batch_norm,
            kernel_initializer=self.conv_initializer,
            activation=None if self.use_batch_norm else activation,
            *args, **kwargs)
        if self.use_batch_norm:
            x = self.batch_norm(x)
            x = activation(x) if activation is not None else x
        return x

    def conv2d_linear_last_bn(self, inputs, *args, **kwargs):
        x = tf.layers.conv2d(
            inputs, data_format=self.data_format,
            use_bias=False,
            kernel_initializer=self.conv_initializer,
            activation=None, *args, **kwargs)
        param_initializers = {
            'moving_mean': tf.zeros_initializer(),
            'moving_variance': tf.ones_initializer(),
            'beta': tf.zeros_initializer(),
        }
        if self.adv_bn_init:
            param_initializers['gamma'] = tf.zeros_initializer()
        else:
            param_initializers['gamma'] = tf.ones_initializer()
        x = self.batch_norm(x, param_initializers=param_initializers)
        return x

    def conv2d_linear(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, None, *args, **kwargs)

    def conv2d(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, self.activation, *args, **kwargs)

    def pad2d(self, inputs, begin, end=None):
        if end is None:
            end = begin
        try:
            _ = begin[1]
        except TypeError:
            begin = [begin, begin]
        try:
            _ = end[1]
        except TypeError:
            end = [end, end]
        if self.data_format == 'channels_last':
            padding = [[0, 0], [begin[0], end[0]], [begin[1], end[1]], [0, 0]]
        else:
            padding = [[0, 0], [0, 0], [begin[0], end[0]], [begin[1], end[1]]]
        return tf.pad(inputs, padding)

    def max_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.max_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def average_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.average_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def dense_linear(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=None)

    def dense(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=self.activation)

    def activate(self, inputs, activation=None):
        activation = activation or self.activation
        return activation(inputs) if activation is not None else inputs

    def batch_norm(self, inputs, **kwargs):
        all_kwargs = dict(self.batch_norm_config)
        all_kwargs.update(kwargs)
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        return tf.contrib.layers.batch_norm(
            inputs, is_training=self.training, data_format=data_format,
            fused=True, **all_kwargs)

    def spatial_average2d(self, inputs):
        shape = inputs.get_shape().as_list()
        if self.data_format == 'channels_last':
            n, h, w, c = shape
            x = tf.reduce_mean(inputs,axis=[1,2])
        else:
            n, c, h, w = shape
            x = tf.reduce_mean(inputs,axis=[2,3])
        n = -1 if n is None else n
        return tf.reshape(x, [n, c])

    def flatten2d(self, inputs):
        x = inputs
        if self.data_format != 'channel_last':
            # Note: This ensures the output order matches that of NHWC networks
            x = tf.transpose(x, [0, 2, 3, 1])
        input_shape = x.get_shape().as_list()
        num_inputs = 1
        for dim in input_shape[1:]:
            num_inputs *= dim
        return tf.reshape(x, [-1, num_inputs], name='flatten')

    def residual2d(self, inputs, network, units=None, scale=1.0, activate=False):
        outputs = network(inputs)
        c_axis = -1 if self.data_format == 'channels_last' else 1
        h_axis = 1 if self.data_format == 'channels_last' else 2
        w_axis = h_axis + 1
        ishape, oshape = [y.get_shape().as_list() for y in [inputs, outputs]]
        ichans, ochans = ishape[c_axis], oshape[c_axis]
        strides = ((ishape[h_axis] - 1) // oshape[h_axis] + 1,
                   (ishape[w_axis] - 1) // oshape[w_axis] + 1)
        with tf.name_scope('residual'):
            if (ochans != ichans or strides[0] != 1 or strides[1] != 1):
                inputs = self.conv2d_linear(inputs, units, 1, strides, 'SAME')
            x = inputs + scale * outputs
            if activate:
                x = self.activate(x)
        return x


def resnet_bottleneck_v1(builder, inputs, depth, depth_bottleneck, stride,
                         basic=False):
    num_inputs = inputs.get_shape().as_list()[1]
    x = inputs
    with tf.name_scope('resnet_v1'):
        if depth == num_inputs:
            if stride == 1:
                shortcut = x
            else:
                shortcut = builder.max_pooling2d(x, 1, stride)
        else:
            shortcut = builder.conv2d_linear(x, depth, 1, stride, 'SAME')
        if basic:
            x = builder.pad2d(x, 1)
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'VALID')
            x = builder.conv2d_linear(x, depth, 3, 1, 'SAME')
        else:
            x = builder.conv2d(x, depth_bottleneck, 1, 1, 'SAME')
            x = builder.conv2d(x, depth_bottleneck, 3, stride, 'SAME')
            # x = builder.conv2d_linear(x, depth,            1, 1,      'SAME')
            x = builder.conv2d_linear_last_bn(x, depth, 1, 1, 'SAME')
        x = tf.nn.relu(x + shortcut)
        return x


def inference_resnet_v1_impl(builder, inputs, layer_counts, basic=False):
    x = inputs
    x = builder.pad2d(x, 3)
    x = builder.conv2d(x, 64, 7, 2, 'VALID')
    x = builder.max_pooling2d(x, 3, 2, 'SAME')
    for i in range(layer_counts[0]):
        x = resnet_bottleneck_v1(builder, x, 256, 64, 1, basic)
    for i in range(layer_counts[1]):
        x = resnet_bottleneck_v1(builder, x, 512, 128, 2 if i == 0 else 1, basic)
    for i in range(layer_counts[2]):
        x = resnet_bottleneck_v1(builder, x, 1024, 256, 2 if i == 0 else 1, basic)
    for i in range(layer_counts[3]):
        x = resnet_bottleneck_v1(builder, x, 2048, 512, 2 if i == 0 else 1, basic)
    return builder.spatial_average2d(x)


def inference_resnet_v1(inputs, data_format='channels_last',
                        training=False, conv_initializer=None, adv_bn_init=True):
    """Deep Residual Networks family of models
    https://arxiv.org/abs/1512.03385
    """
    builder = LayerBuilder(tf.nn.relu, data_format, training, use_batch_norm=True,
                           conv_initializer=conv_initializer, adv_bn_init=adv_bn_init)

    return inference_resnet_v1_impl(builder, inputs, [3, 4, 6, 3])

