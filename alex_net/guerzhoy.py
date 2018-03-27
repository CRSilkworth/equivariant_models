import numpy as np
import tensorflow as tf


def var_wrap(shape, stddev=0.01, bias_init=0):
    weights = tf.get_variable(
        initializer=tf.truncated_normal(shape, stddev=stddev),
        name='weights'
    )

    if not bias_init:
        biases = tf.get_variable(
            initializer=tf.zeros([shape[-1]]),
            name='biases'
        )
    else:
        biases = tf.get_variable(
            initializer=tf.ones([shape[-1]]),
            name='biases'
        )

    return weights, biases


def conv(input, kernel_height, kernel_width, channels_out, stride_height, stride_width, channels_in=None, padding="VALID", bias_init=0, name=None, group=1):
    if channels_in is None:
        channels_in = int(input.get_shape()[-1])

    assert channels_in % group == 0
    assert channels_out % group == 0

    with tf.variable_scope(name):
        shape = [kernel_height, kernel_width, channels_in/group, channels_out]
        kernel, biases = var_wrap(shape, bias_init=bias_init)

    def convolve(input, kernel):
        return tf.nn.conv2d(input, kernel, [1, stride_height, stride_width, 1], padding=padding)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)

    r_conv = tf.reshape(
        tf.nn.bias_add(conv, biases),
        [-1] + conv.get_shape().as_list()[1:]
    )

    return r_conv


def load_weights(session, weights_path, retrain_vars):
    """Load the initial weights from a numpy file
    Does not init the layers that we want to retrain
    Args:
        session: current tensorflow session
        weights_path:
        retrain_vars:
    """
    print "Info: Restoring weights from numpy file: {}".format(weights_path)

    # Load the weights into memory
    weights_dict = np.load(weights_path, encoding='bytes').item()
    # Loop over all layer ops
    for op_name in weights_dict:
        op_name_string = op_name if isinstance(op_name, str) else op_name.decode('utf8')

        # Check if the layer is one of the layers that should be reinitialized
        if op_name_string not in retrain_vars:
            print "  restore: {}".format(op_name_string)

            with tf.variable_scope(op_name_string, reuse=True):
                # Loop over list of weights/biases and assign them to their corresponding tf variable

                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))
                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))

        else:
            print "  skip: {}".format(op_name_string)


class AlexNet(object):
    def __init__(self, images, keep_prob, num_classes, retrain_layer, image_size=(224, 224), weights_path=None):
        self.images = images
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.retrain_layer = retrain_layer
        self.weights_path = weights_path
        self.image_size = image_size
        self.ops = self.create()

    def get_logits(self):
        return self.ops['fc8']

    def load_initial_weights(self, session):
        load_weights(session, self.weights_path, self.retrain_layer)

    def create(self):
        ops = {}
        # conv1
        # kernel_height = 11; kernel_width = 11; channels_out = 96; stride_height = 4; stride_width = 4
        conv1_in = conv(
            input=self.images,
            kernel_height=11,
            kernel_width=11,
            channels_in=3,
            channels_out=96,
            stride_height=4,
            stride_width=4,
            padding="SAME",
            bias_init=0,
            name='conv1',
            group=1
        )

        conv1 = tf.nn.relu(conv1_in, name='conv1')
        ops['conv1'] = conv1

        # lrn1
        # radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(
            conv1,
            depth_radius=2,
            alpha=2e-05,
            beta=0.75,
            bias=1.0,
            name='lrn1'
        )

        ops['lrn1'] = lrn1

        # maxpool1
        # kernel_height = 3; kernel_width = 3; stride_height = 2; stride_width = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(
            lrn1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name='maxpool1'
        )
        ops['maxpool1'] = maxpool1

        # conv2
        # kernel_height = 5; kernel_width = 5; channels_out = 256; stride_height = 1; stride_width = 1; group = 2
        conv2_in = conv(
            input=maxpool1,
            kernel_height=5,
            kernel_width=5,
            # channels_in=96,
            channels_out=256,
            stride_height=1,
            stride_width=1,
            padding="SAME",
            bias_init=1,
            name='conv2',
            group=2
        )
        conv2 = tf.nn.relu(conv2_in, name='conv2')
        ops['conv2'] = conv2

        # lrn2
        # radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(
            conv2,
            depth_radius=2,
            alpha=2e-05,
            beta=0.75,
            bias=1.0,
            name='lrn2'
        )
        ops['lrn2'] = lrn2

        # maxpool2
        # kernel_height = 3; kernel_width = 3; stride_height = 2; stride_width = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(
            lrn2,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name='maxpool2'
        )
        ops['maxpool2'] = maxpool2

        # conv3
        # kernel_height = 3; kernel_width = 3; channels_out = 384; stride_height = 1; stride_width = 1; group = 1
        conv3_in = conv(
            input=maxpool2,
            kernel_height=3,
            kernel_width=3,
            channels_out=384,
            stride_height=1,
            stride_width=1,
            padding="SAME",
            bias_init=0,
            name='conv3',
            group=1
        )
        conv3 = tf.nn.relu(conv3_in, name='conv3')
        ops['conv3'] = conv3

        # conv4
        # kernel_height = 3; kernel_width = 3; channels_out = 384; stride_height = 1; stride_width = 1; group = 2
        conv4_in = conv(
            input=conv3,
            kernel_height=3,
            kernel_width=3,
            # channels_in=384,
            channels_out=384,
            stride_height=1,
            stride_width=1,
            padding="SAME",
            bias_init=1,
            name='conv4',
            group=1
        )
        conv4 = tf.nn.relu(conv4_in, name='conv4')
        ops['conv4'] = conv4

        # conv5
        # kernel_height = 3; kernel_width = 3; channels_out = 256; stride_height = 1; stride_width = 1; group = 2
        conv5_in = conv(
            input=conv4,
            kernel_height=3,
            kernel_width=3,
            # channels_in=384,
            channels_out=256,
            stride_height=1,
            stride_width=1,
            padding="SAME",
            bias_init=5,
            name='conv5',
            group=1
        )
        conv5 = tf.nn.relu(conv5_in, name='conv5')
        ops['conv5'] = conv5

        # maxpool5
        # kernel_height = 3; kernel_width = 3; stride_height = 2; stride_width = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(
            conv5,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='VALID',
            name='maxpool5'
        )
        ops['maxpool5'] = maxpool5
        # fc6
        shape_in = int(np.prod(maxpool5.get_shape()[1:]))
        with tf.variable_scope('fc6'):
            fc6W, fc6b = var_wrap([shape_in, 4096], bias_init=1)
        flattened = tf.reshape(
            maxpool5,
            [-1, shape_in]
        )
        fc6 = tf.nn.relu_layer(flattened, fc6W, fc6b)
        fc6 = tf.nn.dropout(fc6, self.keep_prob)
        ops['fc6'] = fc6

        # fc7
        with tf.variable_scope('fc7'):
            fc7W, fc7b = var_wrap([4096, 4096], bias_init=1)
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        fc7 = tf.nn.dropout(fc7, self.keep_prob)
        ops['fc7'] = fc7

        # fc8
        with tf.variable_scope('fc8'):
            fc8W, fc8b = var_wrap([4096, self.num_classes], bias_init=1)
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        ops['fc8'] = fc8

        return ops
