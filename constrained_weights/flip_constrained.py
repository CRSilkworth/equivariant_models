import tensorflow as tf
import numpy as np


def flip_equivariant_weights(shape, out_axis, in_axis, initializer=None, initializer_kwargs=None):
    if initializer is None:
        initializer = tf.truncated_normal
    if initializer_kwargs is None:
        initializer_kwargs = {}

    in_dim = shape[in_axis]
    out_dim = shape[out_axis]
    total = in_dim * out_dim

    degrees_of_freedom = total/2 if total % 2 == 0 else total/2 + 1
    degrees_of_freedom = np.array([degrees_of_freedom], dtype=np.int32)

    irrelevant_dims = [dim for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]
    irrelevant_axes = [axis for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]

    serialized_shape = np.concatenate([degrees_of_freedom, irrelevant_dims], axis=0)
    serialized_shape = serialized_shape.astype(np.int32)

    serialized = tf.get_variable(
        'weights',
        initializer=initializer(shape=serialized_shape, **initializer_kwargs)
    )
    if (in_dim * out_dim) % 2 == 0:
        serialized = tf.concat([serialized, tf.reverse(serialized, axis=[0])], axis=0)
    else:
        serialized = tf.concat([serialized, tf.reverse(serialized, axis=[0])[1:]], axis=0)

    weights = tf.reshape(serialized, [out_dim, in_dim] + irrelevant_dims)

    old_indices = [out_axis, in_axis] + irrelevant_axes
    new_indices = []
    for index in range(len(shape)):
        new_indices.append(old_indices.index(index))

    weights = tf.transpose(weights, new_indices)

    return weights


def flip_invariant_weights(shape, out_axis, in_axis, initializer=None, initializer_kwargs=None):
    if initializer is None:
        initializer = tf.truncated_normal
    if initializer_kwargs is None:
        initializer_kwargs = {}

    in_dim = shape[in_axis]
    out_dim = shape[out_axis]

    degrees_of_freedom = out_dim/2 if out_dim % 2 == 0 else out_dim/2 + 1
    degrees_of_freedom = np.array([degrees_of_freedom], dtype=np.int32)

    irrelevant_dims = [dim for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]
    irrelevant_axes = [axis for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]

    half_shape = np.concatenate([degrees_of_freedom, in_dim, irrelevant_dims], axis=0)
    half_shape = half_shape.astype(np.int32)

    half_weights = tf.get_variable(
        'weights',
        initializer=initializer(shape=half_shape, **initializer_kwargs)
    )
    if out_dim % 2 == 0:
        weights = tf.concat([half_weights, tf.reverse(half_weights, axis=[0])], axis=0)
    else:
        weights = tf.concat([half_weights, tf.reverse(half_weights, axis=[0])[1:]], axis=0)

    old_indices = [out_axis, in_axis] + irrelevant_axes
    new_indices = []
    for index in range(len(shape)):
        new_indices.append(old_indices.index(index))

    weights = tf.transpose(weights, new_indices)

    return weights


def flip_equivariant_layer(input, shape_out, flip_axis, initializer=None, initializer_kwargs=None, bias_init=0, activation=None):
    shape_in = input.get_shape()[1:]

    shape = np.concatenate([shape_out, shape_in], axis=0)
    # rank_in = len(shape_in)
    rank_out = len(shape_out)

    weights = flip_equivariant_weights(shape, flip_axis,  flip_axis + rank_out, initializer=initializer, initializer_kwargs=initializer_kwargs)

    num_rows = int(np.prod(shape_out))
    num_cols = int(np.prod(shape_in))

    weights_matrix = tf.transpose(tf.reshape(weights, [num_rows, num_cols]))
    flattend_input = tf.contrib.layers.flatten(input)

    # If a zero value was given to bias_init then set initialze them to zero.
    # Otherwise intialize them to one.
    if not bias_init:
        bias_func = tf.zeros
    else:
        bias_func = tf.ones

    biases = tf.get_variable(
        initializer=bias_func([num_rows]),
        name='biases'
    )
    if activation == 'relu':
        return tf.nn.relu_layer(flattend_input, weights_matrix, biases)
    else:
        return tf.nn.xw_plus_b(flattend_input, weights_matrix, biases)
