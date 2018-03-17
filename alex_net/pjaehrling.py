#
# Author: Philipp Jaehrling philipp.jaehrling@gmail.com)
# Influenced by: https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
#
from collections import OrderedDict

import tensorflow as tf
import numpy as np


class Model(object):
    """
    Parent class for multiple CNN model classes
    """
    # These params should be filled for each model
    image_size = 0

    def __init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path):
        """
        Args:
            tensor: tf.placeholder, for the input images
            keep_prob: tf.placeholder, for the dropout rate
            num_classes: int, number of classes of the new dataset
            retrain_layer: list of strings, names of the layers you want to reinitialize
            weights_path: path string, path to the pretrained weights (numpy or checkpoint)
        """
        self.tensor = tensor
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.retrain_layer = retrain_layer
        self.weights_path = weights_path

    def get_final_op(self):
        """Get the net output (final op)

        Returns: the last op containing the log predictions and end_points dict
        """
        raise NotImplementedError("Subclass must implement method")

    def get_endpoints(self):
        """Get an ordered dict with all endpoints

        Returns: ordered endpoints dict
        """
        raise NotImplementedError("Subclass must implement method")

    def get_restore_vars(self):
        """Get a list of tensors, which should be restored
        """
        raise NotImplementedError("Subclass must implement method")

    def get_retrain_vars(self):
        """Get a list of tensors, which should be retrained
        """
        raise NotImplementedError("Subclass must implement method")

    def load_initial_weights(self, session):
        """Load the initial weights
        Args:
            session: current tensorflow session
        """
        raise NotImplementedError("Subclass must implement method")

    def is_layer_trainable(self, layer_name):
        """Return is a layer is trainable or not
        """
        return True if layer_name in self.retrain_layer else False
class AlexNet(Model):
    """
    AlexNet model definition for Tensorflow
    """
    image_size = 227

    def __init__(self, tensor, keep_prob=1.0, num_classes=1000, retrain_layer=[], weights_path='./weights/bvlc_alexnet.npy'):
        # Call the parent class, which will create the graph
        Model.__init__(self, tensor, keep_prob, num_classes, retrain_layer, weights_path)

        # Call the create function to build the computational graph
        self.final, self.endpoints = self.create()

    def get_final_op(self):
        return self.final

    def get_endpoints(self):
        return self.endpoints

    def get_restore_vars(self):
        return [v for v in tf.global_variables() if not v.name.split('/')[0] in self.retrain_layer]

    def get_retrain_vars(self):
        return tf.trainable_variables()

    def load_initial_weights(self, session):
        load_weights(session, self.weights_path, self.retrain_layer)

    def create(self):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.tensor, 11, 11, 96, 4, 4, padding='VALID', name='conv1', trainable=self.is_layer_trainable('conv1'))
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0, name='norm1')
        pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2', trainable=self.is_layer_trainable('conv2'))
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2,alpha=2e-05, beta=0.75, bias=1.0, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3', trainable=self.is_layer_trainable('conv3'))

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4', trainable=self.is_layer_trainable('conv4'))

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5', trainable=self.is_layer_trainable('conv5'))
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        pool5_out  = int(np.prod(pool5.get_shape()[1:])) # 6 * 6 * 256 = 9216
        pool5_flat = tf.reshape(pool5, [-1, pool5_out]) # shape=(image count, 6, 6, 256) -> shape=(image count, 9216)
        fc6        = fc(pool5_flat, 4096, name='fc6', trainable=self.is_layer_trainable('fc6'))
        dropout6   = tf.nn.dropout(fc6, self.keep_prob)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7      = fc(dropout6, 4096, name='fc7', trainable=self.is_layer_trainable('fc7'))
        dropout7 = tf.nn.dropout(fc7, self.keep_prob)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = fc(dropout7, self.num_classes, relu=False, name='fc8', trainable=self.is_layer_trainable('fc8'))

        # add layers to the endpoints dict
        endpoints = OrderedDict()
        endpoints['conv1'] = conv1
        endpoints['pool1'] = pool1
        endpoints['conv2'] = conv2
        endpoints['pool2'] = pool2
        endpoints['conv3'] = conv3
        endpoints['conv4'] = conv4
        endpoints['conv5'] = conv5
        endpoints['pool5'] = pool5
        endpoints['pool5/flat'] = pool5_flat # 9216
        endpoints['fc6'] = fc6 # 4096
        endpoints['fc7'] = fc7 # 4096
        endpoints['fc8'] = fc8 # number of classes

        return fc8, endpoints
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
                        var = tf.get_variable('biases', trainable = False)
                        session.run(var.assign(data))
                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable = False)
                        session.run(var.assign(data))
        else:
            print "  skip: {}".format(op_name_string)

def conv(tensor, filter_height, filter_width, num_filters, stride_y, stride_x, name, trainable = True, padding='SAME', groups=1):
    """
    Wrapper around the tensorflow conv-layer op
    Args:
        tensor:
        filter_height:
        filter_width:
        num_filters:
        stride_y:
        stride_x:
        name:
        padding:
        groups:
    Returns:
    """
    input_channels = int(tensor.get_shape()[-1])
    channels_per_layer = int(input_channels / groups) # In case we split the data for multiple parallel conv-layer
    strides = [1, stride_y, stride_x, 1]
    shape = [filter_height, filter_width, channels_per_layer, num_filters]

    # -> Outputs random values from a truncated normal distribution (with the given standard deviation)
    init_w = tf.truncated_normal(shape, name='weights', dtype=tf.float32, stddev=0.001)
    init_b = tf.zeros([num_filters])

    # tf.nn.conv2d --> Computes a 2-D convolution given 4-D input and filter tensors
    convolve = lambda input, kernel: tf.nn.conv2d(input, kernel, strides=strides, padding=padding)

    with tf.variable_scope(name) as scope:
        # tf.get_variable(...) --> get an existing variable with these parameters or create a new one
        # ... prefixes the name with the current variable scope and performs reuse checks
        weights = tf.get_variable(
            'weights',
            # shape=shape,
            trainable=trainable,
            initializer=init_w
        )

        # Add the convolution
        if groups == 1:
            convolution = convolve(tensor, weights)
        else:
            # In the cases of multiple groups, split inputs & weights and convolve them separately
            input_groups  = tf.split(num_or_size_splits=groups, value=tensor, axis=3)
            weight_groups = tf.split(num_or_size_splits=groups, value=weights, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            convolution = tf.concat(values=output_groups, axis=3)

        # Add biases
        biases = tf.get_variable(
            'biases',
            # shape=[num_filters],
            trainable=trainable,
            initializer=init_b
        )

        # out = tf.reshape(tf.nn.bias_add(convolution, biases), convolution.get_shape().as_list())
        # --> reshape([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        out = tf.nn.bias_add(convolution, biases)

        # Apply relu function --> computes rectified linear: max(features, 0)
        relu = tf.nn.relu(out, name=scope.name)
        return relu


def fc(tensor, num_out, name, trainable = True, relu = True):
    """
    Wrapper around the tensorflow fully connected layer op
    Args:
        tensor:
        num_in:
        num_out:
        name:
        relu:
    Returns:
    """
    num_in = int(tensor.get_shape()[-1])
    init_w = tf.truncated_normal([num_in, num_out], name='weights', dtype=tf.float32, stddev=0.001)
    init_b = tf.ones([num_out]) # tf.zeros([num_out])

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable(
            'weights',
            # shape=[num_in, num_out],
            trainable=trainable,
            initializer=init_w
        )
        # weights = tf.get_variable(
        #     'weights',
        #     shape=[num_in, num_out],
        #     trainable=trainable,
        #     initializer=tf.contrib.layers.xavier_initializer()
        # )
        biases = tf.get_variable(
            'biases',
            # shape=[num_out],
            trainable=trainable,
            initializer=init_b
        )

        # Matrix multiply weights and inputs and add bias
        # act = tf.nn.bias_add(tf.matmul(tensor, weights), biases, name=scope)
        act = tf.nn.xw_plus_b(tensor, weights, biases, name=scope.name)

        if relu is True:
            # Apply relu function --> computes rectified linear: max(features, 0)
            relu = tf.nn.relu(act, name=scope.name)
            return relu
        else:
            return act
