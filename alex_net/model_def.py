# Created on May 31 2018
# Author: CRSilkworth, based on implementation by guerzhoy
"""Model definition of AlexNet."""
import numpy as np
import tensorflow as tf


def var_wrap(shape, stddev=0.01, bias_init=0):
    """Helper function to make defining weights, biases easier."""
    # Define the weights with mean zero
    weights = tf.get_variable(
        initializer=tf.truncated_normal(shape, stddev=stddev),
        name='weights'
    )

    # If a zero value was given to bias_init then set initialze them to zero.
    # Otherwise intialize them to one.
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


def conv(input, kernel_height, kernel_width, channels_out, stride_height, stride_width, channels_in=None, padding="VALID", bias_init=0, group=1, data_format='NHWC'):
    """Helper function to define convolutional layers."""
    # Set the dimension of the channels in and strides parameterself.
    # according to the data format.
    if data_format == 'NHWC':
        channels_dim = 3
        strides = [1, stride_height, stride_width, 1]

    elif data_format == 'NCHW':
        channels_dim = 1
        strides = [1, 1, stride_height, stride_width]

    else:
        raise(ValueError), ("data_format must either be NHWC or NCHW")

    # If no size of channels is was given then infer it from the input.
    if channels_in is None:
        channels_in = int(input.get_shape()[channels_dim])

    # Make sure you're able to split up the groups evenly.
    assert channels_in % group == 0
    assert channels_out % group == 0

    # Create the filter and biases
    shape = [kernel_height, kernel_width, channels_in/group, channels_out]
    kernel, biases = var_wrap(shape, bias_init=bias_init)

    # Define a convolution helper function.
    def convolve(input, kernel):
        return tf.nn.conv2d(input, kernel, strides, padding=padding, data_format=data_format)

    # If there is only one group then convolve the whole input.
    if group == 1:
        conv = convolve(input, kernel)
    else:
        # Otherwise, split the input up into groups along the channels in
        # dimension and the filters up along the channesl out dimension.
        input_groups = tf.split(input, group, channels_dim)
        kernel_groups = tf.split(kernel, group, 3)

        # Separately convolve the two groups.
        output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]

        # Concat them back together along the output channels dimension.
        conv = tf.concat(output_groups, channels_dim)

    # Add the bias
    # r_conv = tf.reshape(
    #     tf.nn.bias_add(conv, biases, data_format=data_format),
    #     [-1] + conv.get_shape().as_list()[1:]
    # )
    r_conv = tf.nn.bias_add(conv, biases, data_format=data_format)

    return r_conv


class AlexNet(object):
    """Class which holds the network architecture, several helper functions and model parameters."""

    def __init__(self, images, keep_prob, num_classes, image_size=(224, 224), data_format='NHWC'):
        """
        Create an AlexNet model. Recieves a tensor of images, feeds it through the network architecture and produces unscaled logits.

        Args:
            images: a rank 4 tensor of of shape [batch_size, image_height, image_width, num_channels].
            keep_prob: The probability a node will be kept during dropout.
            num_classes: The number of possible labels a image can have. Controls number of logits.
            image_size: The size of the inputted images. Should match up with 'images' tensor.
            data_format: The format to put the input into before feeding through the model. NHWC is the inputted format but NCHW is generally faster for convolutions.
        """
        self.images = images
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.image_size = image_size

        self.data_format = data_format

        if data_format == 'NCHW':
            self.images = tf.transpose(self.images, [0, 3, 1, 2])

        self.ops = self.create()

    def get_logits(self):
        """Return the logits operation of the model."""
        return self.ops['fc8']

    def create(self):
        """Create the full network architecture of the model."""
        # Create a dictionary store all the ops that will be returned in.
        ops = {}

        # Define the size and strides of the max pool operations according to
        # the data format.
        if self.data_format == 'NHWC':
            pool_strides = [1, 2, 2, 1]
            pool_ksize = [1, 3, 3, 1]
        elif self.data_format == 'NCHW':
            pool_strides = [1, 1, 2, 2]
            pool_ksize = [1, 1, 3, 3]
        else:
            raise(ValueError), "Invalid data format"

        # CONVOLUTION 1
        with tf.variable_scope('conv1'):
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
                group=1,
                reuse=self.reuse,
                data_format=self.data_format
            )

            conv1 = tf.nn.relu(conv1_in)
            ops['conv1'] = conv1

            lrn1 = tf.nn.local_response_normalization(
                conv1,
                depth_radius=2,
                alpha=2e-05,
                beta=0.75,
                bias=1.0,
            )

            ops['lrn1'] = lrn1

            maxpool1 = tf.nn.max_pool(
                lrn1,
                ksize=pool_ksize,
                strides=pool_strides,
                padding='VALID',
                data_format=self.data_format
            )
            ops['maxpool1'] = maxpool1

        # CONVOLUTION 3
        with tf.variable_scope('conv3'):
            conv2_in = conv(
                input=maxpool1,
                kernel_height=5,
                kernel_width=5,
                channels_out=256,
                stride_height=1,
                stride_width=1,
                padding="SAME",
                bias_init=1,
                group=2,
                reuse=self.reuse,
                data_format=self.data_format
            )
            conv2 = tf.nn.relu(conv2_in)
            ops['conv2'] = conv2

            lrn2 = tf.nn.local_response_normalization(
                conv2,
                depth_radius=2,
                alpha=2e-05,
                beta=0.75,
                bias=1.0,
            )
            ops['lrn2'] = lrn2

            maxpool2 = tf.nn.max_pool(
                lrn2,
                ksize=pool_ksize,
                strides=pool_strides,
                padding='VALID',
                data_format=self.data_format
            )
            ops['maxpool2'] = maxpool2

        # CONVOLUTION 3
        with tf.variable_scope('conv3'):
            conv3_in = conv(
                input=maxpool2,
                kernel_height=3,
                kernel_width=3,
                channels_out=384,
                stride_height=1,
                stride_width=1,
                padding="SAME",
                bias_init=0,
                group=1,
                reuse=self.reuse,
                data_format=self.data_format
            )
            conv3 = tf.nn.relu(conv3_in)
            ops['conv3'] = conv3

        # CONVOLUTION 4
        with tf.variable_scope('conv4'):
            conv4_in = conv(
                input=conv3,
                kernel_height=3,
                kernel_width=3,
                channels_out=384,
                stride_height=1,
                stride_width=1,
                padding="SAME",
                bias_init=1,
                group=1,
                reuse=self.reuse,
                data_format=self.data_format
            )
            conv4 = tf.nn.relu(conv4_in)
            ops['conv4'] = conv4

        # CONVOLUTION 5
        with tf.variable_scope('conv5'):
            conv5_in = conv(
                input=conv4,
                kernel_height=3,
                kernel_width=3,
                channels_out=256,
                stride_height=1,
                stride_width=1,
                padding="SAME",
                bias_init=5,
                group=1,
                reuse=self.reuse,
                data_format=self.data_format
            )
            conv5 = tf.nn.relu(conv5_in)
            ops['conv5'] = conv5

            maxpool5 = tf.nn.max_pool(
                conv5,
                ksize=pool_ksize,
                strides=pool_strides,
                padding='VALID',
                data_format=self.data_format
            )

            if self.data_format == 'NCHW':
                maxpool5 = tf.transpose(maxpool5, [0, 2, 3, 1])
            ops['maxpool5'] = maxpool5

        # FULLY CONNECTED 6
        with tf.variable_scope('fc6', reuse=self.reuse):
            shape_in = int(np.prod(maxpool5.get_shape()[1:]))
            fc6W, fc6b = var_wrap([shape_in, 4096], bias_init=1)
            flattened = tf.reshape(
                maxpool5,
                [-1, shape_in]
            )
            fc6 = tf.nn.relu_layer(flattened, fc6W, fc6b)
            fc6 = tf.nn.dropout(fc6, self.keep_prob)
            ops['fc6'] = fc6

        # FULLY CONNECTED 7
        with tf.variable_scope('fc7', reuse=self.reuse):
            fc7W, fc7b = var_wrap([4096, 4096], bias_init=1)
            fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
            fc7 = tf.nn.dropout(fc7, self.keep_prob)
            ops['fc7'] = fc7

        # FULLY CONNECTED 8
        with tf.variable_scope('fc8', reuse=self.reuse):
            fc8W, fc8b = var_wrap([4096, self.num_classes], bias_init=1)
            fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            ops['fc8'] = fc8

        return ops
