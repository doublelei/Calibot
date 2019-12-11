from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from ops import image_embedding
from ops import inputs as input_ops

class SeeAndWriteModel(object):

    def __init__(self, config, mode, train_inception=False):
        """Basic setup.

        Args:
          config: Object containing configuration parameters.
          mode: "train", "eval" or "inference".
          train_inception: Whether the inception submodel variables are trainable.
        """
        assert mode in ["train", "eval", "inference"]
        self.config = config
        self.mode = mode
        self.train_inception = train_inception

        # Reader for the input data.
        # self.reader = tf.TFRecordReader()

        # To match the "Show and Tell" paper we initialize all variables with a
        # random uniform initializer.
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)

        # A float32 Tensor with shape [batch_size, height, width, channels].
        self.images = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.input_seqs = None

        # An int32 Tensor with shape [batch_size, padded_length].
        self.target_seqs = None

        # An int32 0/1 Tensor with shape [batch_size, padded_length].
        self.input_mask = None

        # A float32 Tensor with shape [batch_size, embedding_size].
        self.image_embeddings = None

        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None

        # Collection of variables from the inception submodel.
        self.inception_variables = []

        # Function to restore the inception submodel from checkpoint.
        self.init_fn = None

        # Global step Tensor.
        self.global_step = None

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def process_image(self, encoded_image):
        """Decodes and processes an image string.

        Args:
          encoded_image: A scalar string Tensor; the encoded image.

        Returns:
          A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return input_ops.process_image(encoded_image,
                                              is_training=self.is_training(),
                                              image_format=self.config.image_format)

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.

        Outputs:
          self.images
          self.input_seqs
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)
        """
        if self.mode == "inference":
            # In inference mode, images and inputs are fed via placeholders.
            image_feed = tf.placeholder(
                dtype=tf.string, shape=[], name="image_feed")
            input_feed = tf.placeholder(dtype=tf.int32,
                                        shape=[None, 6],  # batch_size
                                        name="input_feed")

            # Process image and insert batch dimensions.
            images = tf.expand_dims(input_ops.process_image(image_feed,
                                              is_training=self.is_training(),
                                              image_format=self.config.image_format), 0)
            input_seqs = tf.expand_dims(input_feed, 1)

            # No target sequences or input mask in inference mode.
            target_seqs = None
            input_mask = None
            self.images = images
            self.input_seqs = tf.split(input_seqs, 6, axis = 2)
            self.target_seqs = target_seqs
            self.input_mask = input_mask

        else:
            data_files = []
            for pattern in self.config.input_file_pattern.split(","):
                data_files.extend(tf.gfile.Glob(pattern))
            if not data_files:
                tf.logging.fatal("Found no input files matching %s",
                                 self.config.input_file_pattern)
            else:
                tf.logging.info("Prefetching values from %d files matching %s",
                                len(data_files), self.config.input_file_pattern)
            dataset = tf.data.TFRecordDataset(data_files).map(input_ops.parser).repeat(5000)

            images = dataset.map(lambda image, input_seq, target_seq, indicator: image).batch(
                self.config.batch_size, drop_remainder=True)
            iterator_image = images.make_one_shot_iterator()

            input_seqs = dataset.map(lambda image, input_seq, target_seq, indicator: input_seq).padded_batch(
                self.config.batch_size, padded_shapes=[None, 6], drop_remainder=True)
            iterator_input_seq = input_seqs.make_one_shot_iterator()

            target_seqs = dataset.map(lambda image, input_seq, target_seq, indicator: target_seq).padded_batch(
                self.config.batch_size, padded_shapes=[None, 6], drop_remainder=True)
            iterator_target_seq = target_seqs.make_one_shot_iterator()

            indicators = dataset.map(lambda image, input_seq, target_seq, indicator: indicator).padded_batch(
                self.config.batch_size, padded_shapes=[None], drop_remainder=True)
            iterator_indicator = indicators.make_one_shot_iterator()

            self.images = iterator_image.get_next()
            self.input_seqs = tf.split(iterator_input_seq.get_next(), 6, axis = 2)
            self.target_seqs = iterator_target_seq.get_next()
            self.input_mask = iterator_indicator.get_next()

    def build_seq_embeddings(self):
        """Builds the input sequence embeddings.

        Inputs:
          self.input_seqs

        Outputs:
          self.seq_embeddings
        """
        self.seq_embeddings = [tf.one_hot(self.input_seqs[i], depth = self.config.ranges[i], on_value = 1.0, off_value = 0.0) for i in range(6)]

        if self.mode == "inference":
            self.seq_embeddings = [tf.squeeze(i, [1, 2]) for i in self.seq_embeddings]
        else: 
            self.seq_embeddings = [tf.squeeze(i, [2]) for i in self.seq_embeddings]

    def build_image_embeddings(self):
        """Builds the image model subgraph and generates image embeddings.

        Inputs:
          self.images

        Outputs:
          self.image_embeddings
        """
        inception_output = image_embedding.inception_v3(
            self.images,
            trainable=self.train_inception,
            is_training=self.is_training())
        self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

        # Map inception output into embedding space.

        image_embedding_1 = tf.contrib.layers.fully_connected(
            inputs=inception_output,
            num_outputs=self.config.embedding_size,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope="image_embedding_1")
        image_embeddings = [tf.contrib.layers.fully_connected(
            inputs=image_embedding_1,
            num_outputs=self.config.ranges[i],
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope="image_embedding_2_{}".format(i)) for i in range(6)]
        self.image_embeddings = image_embeddings
    
    def build_model(self):
        """Builds the model.

        Inputs:
          self.image_embeddings
          self.seq_embeddings
          self.target_seqs (training and eval only)
          self.input_mask (training and eval only)

        Outputs:
          self.total_loss (training and eval only)
          self.target_cross_entropy_losses (training and eval only)
          self.target_cross_entropy_loss_weights (training and eval only)
        """
        # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
        # modified LSTM in the "Show and Tell" paper has no biases and outputs
        # new_c * sigmoid(o).
        lstm_cells= [tf.nn.rnn_cell.LSTMCell(
            name='basic_lstm_cell_{}'.format(i), num_units=self.config.ranges[i], state_is_tuple=True) for i in range(6)]
        
        if self.mode == "train":
            lstm_cells= [tf.contrib.rnn.DropoutWrapper(
                lstm_cell,
                input_keep_prob=self.config.lstm_dropout_keep_prob,
                output_keep_prob=self.config.lstm_dropout_keep_prob) for lstm_cell in lstm_cells]

        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            # Feed the image embeddings to set the initial LSTM state.
            zero_states = [lstm_cells[i].zero_state(
                batch_size=self.image_embeddings[i].get_shape()[0], dtype=tf.float32) for i in range(6)]

            initial_states = [lstm_cells[i](self.image_embeddings[i], zero_states[i])[1] for i in range(6)]

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()

            if self.mode == "inference":
                # In inference mode, use concatenated states for convenient feeding and
                # fetching.
                for i in range(6):
                    tf.concat(axis=1, values=initial_states[i], name="initial_state_{}".format(i))

                # Placeholder for feeding a batch of concatenated states.

                state_feeds = [tf.placeholder(dtype=tf.float32, shape=[None, self.config.ranges[i]*2], name="state_feed_{}".format(i)) for i in range(6)]

                state_tuples = [tf.split(
                    value=state_feeds[i], num_or_size_splits=2, axis=1) for i in range(6)]

                # Run a single LSTM step.
                tem = [lstm_cells[i](inputs=self.seq_embeddings[i], state=state_tuples[i]) for i in range(6)]
                lstm_outputs = [i[0] for i in tem]
                state_tuples = [i[1] for i in tem]

                # Concatentate the resulting state.
                for i in range(6):
                    tf.concat(axis=1, values=state_tuples[i], name="state_{}".format(i))

                pass
            else:
                # Run the batch of sequence embeddings through the LSTM.
                sequence_length = tf.reduce_sum(self.input_mask, 1)
                lstm_outputs= [tf.nn.dynamic_rnn(cell=lstm_cells[i],
                                                    inputs=self.seq_embeddings[i],
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_states[i],
                                                    dtype=tf.float32,
                                                    scope=lstm_scope)[0] for i in range(6)]
                
        # Stack batches vertically.
        lstm_outputs = [tf.reshape(lstm_outputs[i], [-1, lstm_cells[i].output_size]) for i in range(6)]

        if self.mode == "inference":
            for i in range(6):
                tf.nn.softmax(lstm_outputs[i], name="softmax_{}".format(i))
        else:
            
            targets = tf.split(self.target_seqs, 6, axis = 2)
            targets = [tf.reshape(targets[i], [-1]) for i in range(6)]
            weights = tf.reshape(tf.to_int64(self.input_mask), [-1])
            outputs = [tf.argmax(lstm_outputs[i], axis=1) for i in range(6)]

            # Compute losses.
            losses = [tf.square(targets[i] - outputs[i]) for i in range(6)]
            batch_loss = [tf.multiply(losses[i], weights) for i in range(6)]
            batch_loss = tf.to_float(tf.div(tf.reduce_sum(batch_loss), tf.reduce_sum(weights)))
            tf.losses.add_loss(batch_loss)
            total_loss = tf.losses.get_total_loss()

            # Add summaries.
            tf.summary.scalar("losses/batch_loss", batch_loss)
            tf.summary.scalar("losses/total_loss", total_loss)
            for var in tf.trainable_variables():
                tf.summary.histogram("parameters/" + var.op.name, var)

            self.total_loss = total_loss
            self.target_cross_entropy_losses = losses  # Used in evaluation.
            # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

        self.global_step = global_step

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_image_embeddings()
        self.build_seq_embeddings()
        self.build_model()
        self.setup_global_step()
