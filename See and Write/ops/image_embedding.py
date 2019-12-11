# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

slim = tf.contrib.slim

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)


def inception(inputs,
                      final_endpoint='Mixed_7a',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception model from http://arxiv.org/abs/1512.00567.

  Constructs an Inception v3 network from inputs to the given final endpoint.
  This method can construct the network up to the final inception block
  Mixed_7c.

  Note that the names of the layers in the paper do not correspond to the names
  of the endpoints registered by this function although they build the same
  network.

  Here is a mapping from the old_names to the new names:
  Old name          | New name
  =======================================
  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3
  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d
  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e
  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    scope: Optional variable_scope.

  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with variable_scope.variable_scope(scope, 'InceptionV3', [inputs]):
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
        stride=1,
        padding='VALID'):
      # 298 x 298 x 3
      end_point = 'Conv2d_1a_3x3'
      net = layers.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 149 x 149 x 32
      end_point = 'MaxPool_3a_3x3'
      net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 74 x 74 x 32
      end_point = 'Conv2d_3b_1x1'
      net = layers.conv2d(net, depth(64), [1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 73 x 73 x 64.
      end_point = 'MaxPool_5a_3x3'
      net = layers_lib.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points
      # 36 x 36 x 64.

      # Inception blocks
    with arg_scope(
        [layers.conv2d, layers_lib.max_pool2d, layers_lib.avg_pool2d],
        stride=1,
        padding='SAME'):
      # mixed: 36 x 36 x 160.
      end_point = 'Mixed_5b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(32), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_1: 36 x 36 x 192.
      end_point = 'Mixed_5c'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(64), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_3: 17 x 17 x 256.
      end_point = 'Mixed_6a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net,
              depth(128), [4, 4],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers_lib.max_pool2d(
              net, [4, 4], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = array_ops.concat([branch_0, branch_1], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed4: 17 x 17 x 256.
      end_point = 'Mixed_6b'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(48), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(64), [7, 1], scope='Conv2d_0c_7x1')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_2 = layers.conv2d(
              branch_2, depth(128), [1, 1], scope='Conv2d_0b_1x1')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

      # mixed_8: 8 x 8 x 640.
      end_point = 'Mixed_7a'
      with variable_scope.variable_scope(end_point):
        with variable_scope.variable_scope('Branch_0'):
          branch_0 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = layers.conv2d(
              branch_0,
              depth(192), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_1'):
          branch_1 = layers.conv2d(
              net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
          branch_1 = layers.conv2d(
              branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
          branch_1 = layers.conv2d(
              branch_1,
              depth(192), [3, 3],
              stride=2,
              padding='VALID',
              scope='Conv2d_1a_3x3')
        with variable_scope.variable_scope('Branch_2'):
          branch_2 = layers_lib.max_pool2d(
              net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
        net = array_ops.concat([branch_0, branch_1, branch_2], 3)
      end_points[end_point] = net
      if end_point == final_endpoint:
        return net, end_points

    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net
