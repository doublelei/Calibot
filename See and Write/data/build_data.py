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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from datetime import datetime
import os.path
import random
import sys


import numpy as np
from six.moves import xrange
import tensorflow as tf


tf.flags.DEFINE_string(
    "output_dir", r"data/TFRecord", "Output data directory.")
tf.flags.DEFINE_integer("train_shards", 10,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 3,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 2,
                        "Number of shards in testing TFRecord files.")
tf.flags.DEFINE_string("start_position", "0",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_position", "360",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_integer("num_of_data", 896, "Number of data")
tf.flags.DEFINE_float("multiplier", 0.25, "multiplier")
FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])

max_degree = [0, 0, 0, 0, 0, 0]
min_degree = [2048] * 6

not_worked = [358, 553]
a = set()

class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=1)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    try:
        tem = tf.train.BytesList(value=[value])
    except:
        tem = tf.train.BytesList(value=[value.encode()])
    return tf.train.Feature(bytes_list=tem)

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_sequence_example(data, decoder):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
      image: An ImageMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.

    Returns:
      A SequenceExample proto.
    """
    with tf.gfile.GFile(data[1], "rb") as f:
        encoded_image = f.read()
    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % data[1])
        return

    context = tf.train.Features(feature={
        "image/image_id": _int64_feature(data[0]),
        "image/data": _bytes_feature(encoded_image),
    })

    trajectoires = process_trajectory(data[2])

    feature_lists = tf.train.FeatureLists(feature_list={
        "image/trajectories": _int64_feature_list(trajectoires)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example

def process_trajectory(filename):
    def bytes_to_angle(bytes):
        result = []
        for i in range(6):
            result.append(int((int(bytes[2*i]) + int(bytes[2*i+1]) * 256)*FLAGS.multiplier) + 2)
        return result


    with open(filename, 'r') as f:
        data = f.readlines()
        assert len(data) % 13 == 0
        trajectories = [1, 1, 1, 1, 1, 1]
        global max_degree
        global min_degree
        for i in range(len(data)//13):
            tem = bytes_to_angle(data[i*13+1:(i+1)*13])
            max_degree = [max([max_degree[i], tem[i]]) for i in range(6)]
            min_degree = [min([min_degree[i], tem[i]]) for i in range(6)]
            trajectories.extend(bytes_to_angle(data[i*13+1:(i+1)*13]))
        trajectories.extend([0, 0, 0, 0, 0, 0])

        for i in trajectories:
            a.add(i)

    return trajectories

def _process_image_files(ranges, name, metadata, decoder, num_shards):
    """Processes and saves a subset of metadata as TFRecord files in one thread.

    Args:
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      metadata: List of ImageMetadata.
      decoder: An ImageDecoder object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).

    shard_ranges = np.linspace(ranges[0], ranges[1],
                               num_shards + 1).astype(int)
    counter = 0

    for shard in xrange(num_shards):
        # Generate a sharded version of the file name, e.g. 'train-002-of-010'
        output_filename = "%s-%.3d-of-%.3d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        metadata_in_shard = np.arange(
            shard_ranges[shard], shard_ranges[shard + 1], dtype=int)
        for i in metadata_in_shard:
            data = metadata[i]

            sequence_example = _to_sequence_example(data, decoder)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s : Processed %d of %d items." %
                      (datetime.now(), counter, ranges[1]))
                sys.stdout.flush()

        writer.close()
        print("%s : Wrote %d image-caption pairs to %s" %
              (datetime.now(), shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s : Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), counter, num_shards))
    sys.stdout.flush()

def _process_dataset(name, metadata, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      num_shards: Integer number of shards for the output files.
    """

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(0)
    random.shuffle(metadata)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    ranges = [0, len(metadata)]

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()
    _process_image_files(ranges, name, metadata, decoder, num_shards)

    # Wait for all the threads to terminate.
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(metadata), name))

def _load_and_process_metadata(image_dir, trajectory_dir):
    """Loads image metadata from a JSON file and processes the captions.

    Args:
      captions_file: JSON file containing caption annotations.
      image_dir: Directory containing the image files.

    Returns:
      A list of ImageMetadata.
    """
    metadata = []
    for id in range(FLAGS.num_of_data):
        if (id not in not_worked):
            img_filename = os.path.join(image_dir, 'IMG_{0:0>4}.jpg'.format(id))
            traj_filename = os.path.join(trajectory_dir, str(id) + '.txt')
            metadata.append([id, img_filename, traj_filename])

    return metadata

def main(unused_argv):

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    dataset = _load_and_process_metadata(r'data/Images',
                                         r'data/Trajectories')

    # Redistribute the MSCOCO data as follows:
    # train_cutoff = int(0.95 * len(dataset))
    # val_cutoff = int(0.90 * len(dataset))
    # train_dataset = dataset[:train_cutoff]
    # val_dataset = dataset[train_cutoff:val_cutoff]
    # test_dataset = dataset[val_cutoff:]

    _process_dataset("train", dataset, FLAGS.train_shards)
    # _process_dataset("val", val_dataset, FLAGS.val_shards)
    # _process_dataset("test", test_dataset, FLAGS.test_shards)
    print(max_degree)
    print(min_degree)
    print(len(a))
if __name__ == "__main__":
    tf.app.run()
