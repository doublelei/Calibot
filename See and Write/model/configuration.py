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

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""
    # File pattern of sharded TFRecord file containing SequenceExample protos.
    # Must be provided in training and evaluation modes.
    self.input_file_pattern = None

    # Image format ("jpeg" or "png").
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 2300
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer captions.
    self.caption_feature_name = "image/caption_ids"

    # Batch size.
    self.batch_size = 64

    # Dimensions of Inception v3 input images.
    self.image_height = 298
    self.image_width = 298

    # Scale used to initialize model variables.
    self.initializer_scale = 0.001

    self.lower = [250, 170, 285, 0, 225, 0]
    self.ranges = [503, 688, 285, 1027, 578, 1027]
    self.embedding_size = 512

    # If < 1.0, the dropout keep probability applied to LSTM variables.
    self.lstm_dropout_keep_prob = 0.8


class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    # Number of examples per epoch of training data.
    self.num_examples_per_epoch = 2000

    # Optimizer for training the model.
    self.optimizer = "Adam"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 0.0001
    self.learning_rate_decay_factor = 0.05
    self.num_epochs_per_decay = 10.0

    # Learning rate when fine tuning the Inception v3 parameters.
    self.train_inception_learning_rate = 0.001

    # If not None, clip gradients to this value.
    self.clip_gradients = 10.0

    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5
