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
"""Class for generating captions from an image-to-text model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import math

import tensorflow as tf
import numpy as np

class TrajectoryGenerator(object):
    """Class to generate captions from an image-to-text model."""

    def __init__(self,
                 model,
                 max_trajectory_length=100,
                 length_normalization_factor=0.0):
        """Initializes the generator.

        Args:
          model: Object encapsulating a trained image-to-text model. Must have
            methods feed_image() and inference_step(). For example, an instance of
            InferenceWrapperBase.
          max_caption_length: The maximum caption length before stopping the search.
          length_normalization_factor: If != 0, a number x such that captions are
            scored by logprob/length^x, rather than logprob. This changes the
            relative scores of captions depending on their lengths. For example, if
            x > 0 then longer captions will be favored.
        """
        self.model = model
        self.max_trajectory_length = max_trajectory_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, sess, encoded_image):
        """Runs beam search caption generation on a single image.

        Args:
          sess: TensorFlow Session object.
          encoded_image: An encoded image string.

        Returns:
          A list of Caption sorted by descending score.
        """
        # Feed in the image to get the initial state.
        initial_state = self.model.feed_image(sess, encoded_image)

        trajectories = [[1, 1, 1, 1, 1, 1]]
        states = initial_state

        for _ in range(self.max_trajectory_length - 1):
            input_feed = np.array([trajectories[-1]])
            state_feed = states[0]
            softmaxs, states = self.model.inference_step(sess, input_feed, state_feed)
            softmaxs = [np.argsort(-softmaxs[i]) for i in range(6)]
            # if _ == 0:
            trajectory = [int(softmaxs[i][0][0]) for i in range(6)]

            # else:
            #   for i in range(6):
            #     for index in softmaxs[i][0]:
            #       if abs(index - trajectories[-1][i]) <= 10:
            #         trajectory[i] = index

            if 0 in trajectory or 1 in trajectory:
                break
            trajectories.append([trajectory[i] for i in range(6)])

        return trajectories
