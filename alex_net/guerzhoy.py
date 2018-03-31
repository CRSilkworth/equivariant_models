# Created on May 31 2018
# Author: CRSilkworth, based on implementation by guerzhoy
""" Model definition of AlexNet."""
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


def conv(input, kernel_height, kernel_width, channels_out, stride_height, stride_width, channels_in=None, padding="VALID", bias_init=0, name=None, group=1, reuse=False, data_format='NHWC'):
    if data_format == 'NHWC':
        channels_dim = 3
        strides = [1, stride_height, stride_width, 1]

    elif data_format == 'NCHW':
        channels_dim = 1
        strides = [1, 1, stride_height, stride_width]

    else:
        raise(ValueError), ("data_format must either be NHWC or NCHW")

    if channels_in is None:
        channels_in = int(input.get_shape()[channels_dim])

    assert channels_in % group == 0
    assert channels_out % group == 0

    with tf.variable_scope(name, reuse=reuse):
        shape = [kernel_height, kernel_width, channels_in/group, channels_out]
        kernel, biases = var_wrap(shape, bias_init=bias_init)

    def convolve(input, kernel):
        return tf.nn.conv2d(input, kernel, strides, padding=padding, data_format=data_format)

    if group == 1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, channels_dim)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, channels_dim)

    r_conv = tf.reshape(
        tf.nn.bias_add(conv, biases, data_format=data_format),
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

        print "  restore: {}".format(op_name_string)
        with tf.variable_scope(op_name_string, reuse=True):
            # Loop over list of weights/biases and assign them to their corresponding tf variable

            for data in weights_dict[op_name]:
                # Biases
                if len(data.shape) == 1:
                    var = tf.get_variable('biases')
                    session.run(var.assign(data))
                # Weights
                else:
                    var = tf.get_variable('weights')
                    session.run(var.assign(data))



class AlexNet(object):
    def __init__(self, images, keep_prob, num_classes, retrain_layer=None, image_size=(224, 224), weights_path=None, reuse=False, data_format='NHWC'):
        self.images = images
        self.keep_prob = keep_prob
        self.num_classes = num_classes

        self.retrain_layer = retrain_layer
        if retrain_layer is None:
            self.retrain_layer = []
        self.weights_path = weights_path
        self.image_size = image_size
        self.reuse = reuse
        self.data_format = data_format

        self.data_format = data_format
        if data_format == 'NCHW':
            self.images = tf.transpose(self.images, [0, 3, 1, 2])

        self.ops = self.create()

    def get_logits(self):
        return self.ops['fc8']

    def load_initial_weights(self, session):
        load_weights(session, self.weights_path, self.retrain_layer)

    def create(self):
        ops = {}
        if self.data_format == 'NHWC':
            pool_strides = [1, 2, 2, 1]
            pool_ksize = [1, 3, 3, 1]
        elif self.data_format == 'NCHW':
            pool_strides = [1, 1, 2, 2]
            pool_ksize = [1, 1, 3, 3]
        else:
            raise(ValueError), "Invalid data format"

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
            group=1,
            reuse=self.reuse,
            data_format=self.data_format
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
            ksize=pool_ksize,
            strides=pool_strides,
            padding='VALID',
            name='maxpool1',
            data_format=self.data_format
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
            group=2,
            reuse=self.reuse,
            data_format=self.data_format
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
            ksize=pool_ksize,
            strides=pool_strides,
            padding='VALID',
            name='maxpool2',
            data_format=self.data_format
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
            group=1,
            reuse=self.reuse,
            data_format=self.data_format
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
            group=1,
            reuse=self.reuse,
            data_format=self.data_format
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
            group=1,
            reuse=self.reuse,
            data_format=self.data_format
        )
        conv5 = tf.nn.relu(conv5_in, name='conv5')
        ops['conv5'] = conv5

        # maxpool5
        # kernel_height = 3; kernel_width = 3; stride_height = 2; stride_width = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(
            conv5,
            ksize=pool_ksize,
            strides=pool_strides,
            padding='VALID',
            name='maxpool5',
            data_format=self.data_format
        )
        ops['maxpool5'] = maxpool5

        if self.data_format == 'NCHW':
            maxpool5 = tf.transpose(maxpool5, [0, 2, 3, 1])

        # fc6
        shape_in = int(np.prod(maxpool5.get_shape()[1:]))
        with tf.variable_scope('fc6', reuse=self.reuse):
            fc6W, fc6b = var_wrap([shape_in, 4096], bias_init=1)
        flattened = tf.reshape(
            maxpool5,
            [-1, shape_in]
        )
        fc6 = tf.nn.relu_layer(flattened, fc6W, fc6b)
        fc6 = tf.nn.dropout(fc6, self.keep_prob)
        ops['fc6'] = fc6

        # fc7
        with tf.variable_scope('fc7', reuse=self.reuse):
            fc7W, fc7b = var_wrap([4096, 4096], bias_init=1)
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
        fc7 = tf.nn.dropout(fc7, self.keep_prob)
        ops['fc7'] = fc7

        # fc8
        with tf.variable_scope('fc8', reuse=self.reuse):
            fc8W, fc8b = var_wrap([4096, self.num_classes], bias_init=1)
        fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
        ops['fc8'] = fc8

        return ops
