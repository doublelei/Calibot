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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from model import configuration
from utils import inference_wrapper_see
from utils import trajectory_generator


FLAGS = tf.flags.FLAGS


zi = r'永'

tf.flags.DEFINE_string("checkpoint_path", r"model2",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
# stroke_sperate/{}/?.png
tf.flags.DEFINE_string("input_files", r"data/Images/IMG_0000.jpg".format(zi),
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def angle_to_byte(angle):
  result = []
  for i in range(6):
      pos1, pos2 = divmod(int(angle[i]*2), 256)
      result.append(str(pos2) + '\r\n')
      result.append(str(pos1) + '\r\n')
  return result


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper_see.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = trajectory_generator.TrajectoryGenerator(model)

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      results = generator.beam_search(sess, image)
      savename = filename.replace('png', 'txt').replace(
          'stroke_sperate', 'Trajectories_Target')
      for i in results:
          print(i)
      print(len(results))
      with open(savename, 'w') as f:
          for i in range(1, len(results)):
            f.write('Frame {}\r\n'.format(i))
            for j in angle_to_byte(results[i]):
                f.write(j)



if __name__ == "__main__":
  tf.app.run()
