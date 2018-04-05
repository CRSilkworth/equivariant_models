import tensorflow as tf

def flip_equivariant(shape, out_axis, in_axis, initializer=None, initializer_kwargs=None):
    if initializer is None:
        initializer = tf.truncated_normal
    if initializer_kwargs is None:
        initializer_kwargs = {}

    in_dim = shape[in_axis]
    out_dim = shape[out_axis]
    max_dim = max(in_dim, out_dim)

    degrees_of_freedom = int(max_dim * max_dim/2. + 0.5)

    irrelevant_dims = [dim for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]
    irrelevant_axes = [axis for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]

    serialized_shape = [degrees_of_freedom] + irrelevant_dims

    serialized = tf.get_variable(
        'weights',
        initializer=initializer(shape=serialized_shape, **initializer_kwargs)
    )
    if max_dim % 2 == 0:
        serialized = tf.concat([serialized, tf.reverse(serialized, axis=[0])], axis=0)
    else:
        serialized = tf.concat([serialized, tf.reverse(serialized, axis=[0])[1:]], axis=0)

    weights = tf.reshape(serialized, [max_dim, max_dim] + irrelevant_dims)
    if in_dim > out_dim:
        weights = weights[:out_dim]
    elif out_dim > in_dim:
        weights = weights[:, :in_dim]

    old_indices = [out_axis, in_axis] + irrelevant_axes
    new_indices = []
    for index in range(len(shape)):
        new_indices.append(old_indices.index(index))

    weights = tf.transpose(weights, new_indices)
    # weights = tf.transpose(weights, [])

    return weights
