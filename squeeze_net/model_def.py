"""avoroshilov implementation"""
import tensorflow as tf
import numpy as np
import scipy
def _pool_layer(net, name, input, pooling, size=(2, 2), stride=(3, 3), padding='SAME'):
    if pooling == 'avg':
        x = tf.nn.avg_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    else:
        x = tf.nn.max_pool(input, ksize=(1, size[0], size[1], 1), strides=(1, stride[0], stride[1], 1),
                padding=padding)
    net[name] = x
    return x

def fire_cluster(x, weights_dict, cluster_name):

    # central - squeeze
    weights, biases = weights_dict[cluster_name + '/squeeze1x1']
    conv = tf.nn.conv2d(x, weights, strides=(1, 1, 1, 1), padding='VALID')
    x_i = tf.nn.bias_add(conv, biases)
    x_i = tf.nn.relu(x_i)

    # left - expand 1x1
    weights, biases = weights_dict[cluster_name + '/expand1x1']
    conv = tf.nn.conv2d(x_i, weights, strides=(1, 1, 1, 1), padding='VALID')
    x_l = tf.nn.bias_add(conv, biases)
    x_l = tf.nn.relu(x_l)

    # right - expand 3x3
    weights, biases = weights_dict[cluster_name + '/expand3x3']
    conv = tf.nn.conv2d(x_i, weights, strides=(1, 1, 1, 1), padding='SAME')
    x_r = tf.nn.bias_add(conv, biases)
    x_r = tf.nn.relu(x_r)

    # concatenate expand 1x1 (left) and expand 3x3 (right)
    y = tf.concat([x_l, x_r], 3)

    return y


def create_initial_weights():
    weights_dict = {}
    weights_dict['conv1'] = create_weights_biases((3, 3, 3, 64))
    weights_dict['fire2/expand1x1'] = create_weights_biases((1, 1, 16, 64))
    weights_dict['fire2/expand3x3'] = create_weights_biases((3, 3, 16, 64))
    weights_dict['fire2/squeeze1x1'] = create_weights_biases((1, 1, 64, 16))
    weights_dict['fire3/expand1x1'] = create_weights_biases((1, 1, 16, 64))
    weights_dict['fire3/expand3x3'] = create_weights_biases((3, 3, 16, 64))
    weights_dict['fire3/squeeze1x1'] = create_weights_biases((1, 1, 128, 16))
    weights_dict['fire4/expand1x1'] = create_weights_biases((1, 1, 32, 128))
    weights_dict['fire4/expand3x3'] = create_weights_biases((3, 3, 32, 128))
    weights_dict['fire4/squeeze1x1'] = create_weights_biases((1, 1, 128, 32))
    weights_dict['fire5/expand1x1'] = create_weights_biases((1, 1, 32, 128))
    weights_dict['fire5/expand3x3'] = create_weights_biases((3, 3, 32, 128))
    weights_dict['fire5/squeeze1x1'] = create_weights_biases((1, 1, 256, 32))
    weights_dict['fire6/expand1x1'] = create_weights_biases((1, 1, 48, 192))
    weights_dict['fire6/expand3x3'] = create_weights_biases((3, 3, 48, 192))
    weights_dict['fire6/squeeze1x1'] = create_weights_biases((1, 1, 256, 48))
    weights_dict['fire7/expand1x1'] = create_weights_biases((1, 1, 48, 192))
    weights_dict['fire7/expand3x3'] = create_weights_biases((3, 3, 48, 192))
    weights_dict['fire7/squeeze1x1'] = create_weights_biases((1, 1, 384, 48))
    weights_dict['fire8/expand1x1'] = create_weights_biases((1, 1, 64, 256))
    weights_dict['fire8/expand3x3'] = create_weights_biases((3, 3, 64, 256))
    weights_dict['fire8/squeeze1x1'] = create_weights_biases((1, 1, 384, 64))
    weights_dict['fire9/expand1x1'] = create_weights_biases((1, 1, 64, 256))
    weights_dict['fire9/expand3x3'] = create_weights_biases((3, 3, 64, 256))
    weights_dict['fire9/squeeze1x1'] = create_weights_biases((1, 1, 512, 64))
    weights_dict['conv10'] = create_weights_biases((1, 1, 512, 1000))

    return weights_dict


def squeeze_net_model(image, weights_func=create_initial_weights, weights_func_kwargs=None, is_training=False):

    x_i = tf.cast(image, tf.float32)

    end_points = {}

    if weights_func_kwargs is None:
        weights_func_kwargs = {}

    weights_dict = weights_func(**weights_func_kwargs)
    # CONV 1
    weights, biases = weights_dict['conv1']
    conv = tf.nn.conv2d(x_i, weights, strides=(1, 2, 2, 1), padding='VALID')
    x_i = tf.nn.bias_add(conv, biases)
    x_i = tf.nn.relu(x_i)
    x_1 = tf.nn.max_pool(x_i, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

    # FIRE 2 + 3
    x_2 = fire_cluster(x_1, weights_dict, 'fire2')
    x_3 = fire_cluster(x_2, weights_dict, 'fire3')
    x_3 = tf.nn.max_pool(x_3, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

    # FIRE 2 + 3
    x_4 = fire_cluster(x_3, weights_dict, 'fire4')
    x_5 = fire_cluster(x_4, weights_dict, 'fire5')
    x_5 = tf.nn.max_pool(x_5, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

    # FIRE 6 - 9
    x_6 = fire_cluster(x_5, weights_dict, 'fire6')
    x_7 = fire_cluster(x_6, weights_dict, 'fire7')
    x_8 = fire_cluster(x_7, weights_dict, 'fire8')
    x_9 = fire_cluster(x_8, weights_dict, 'fire9')

    # dropout
    keep_prob = tf.cond(is_training, lambda: 0.5, lambda: 1.)
    x_9 = tf.nn.dropout(x_9, keep_prob)

    # CONV 10
    weights, biases = weights_dict['conv10']
    conv = tf.nn.conv2d(x_9, weights, strides=(1, 1, 1, 1), padding='VALID')
    x_9 = tf.nn.bias_add(conv, biases)
    x_10 = tf.nn.relu(x_9)

    # AVE POOL
    x_f = tf.nn.avg_pool(x_10, ksize=(1, 13, 13, 1), strides=(1, 1, 1, 1), padding='VALID')
    x_f = tf.squeeze(x_f)

    return x_f, end_points


def load_mat_weights(data_path):
    print "LOADING WEIGHTS"
    weights_raw = scipy.io.loadmat(data_path)

    # Converting to needed type
    weights = {}
    for name in weights_raw:
        weights[name] = []
        # skipping '__version__', '__header__', '__globals__'
        if name[0:2] != '__':
            kernels, bias = weights_raw[name][0]
            weights[name].append(tf.constant(kernels.astype(np.float32)))
            weights[name].append(tf.constant(bias.astype(np.float32).reshape(-1)))

    return weights


def create_weights_biases(shape):
    weights = tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32))
    biases = tf.Variable(tf.zeros(shape=[shape[-1]], dtype=tf.float32))

    return weights, biases
