import tensorflow as tf


def rgb_distortion(image, rgb_eigenvectors, rgb_eigenvalues, stddev=0.1, to_run=None):
    height = image.shape[0]
    width = image.shape[1]

    distortions = []
    for color_num in xrange(3):
        eigenvector = rgb_eigenvectors[color_num]
        eigenvalue = tf.sqrt(rgb_eigenvalues[color_num])

        random = tf.truncated_normal([], stddev=stddev)

        distortions.append(random * eigenvalue * eigenvector)

    # Stack them into a single tensor, and add them together.
    distortion = tf.reduce_sum(
        tf.stack(distortions, axis=0),
        axis=0
    )

    to_run['distortion'] = distortion

    # Add the distortion and return
    return image + distortion


def add_horizontal_flip_to_batch(images, labels, flip_axis=2, to_run=None):
    # Flip along the width axis, stack with the original along the batch dim
    # and return.
    flipped = tf.reverse(images, axis=flip_axis)
    labels = tf.tile(labels, multiples=[2])
    return tf.stack([images, flipped], axis=0), labels


def add_five_crops_to_batch(images, labels, crop_image_size, to_run=None):
    # Four corner crops
    crop_1 = images[:, :crop_image_size[0], :crop_image_size[1], :]
    crop_2 = images[:, -crop_image_size[0]:, :crop_image_size[1], :]
    crop_3 = images[:, :crop_image_size[0], :-crop_image_size[1], :]
    crop_4 = images[:, -crop_image_size[0]:, :-crop_image_size[1], :]

    # Center crop
    crop_5 = tf.image.resize_image_with_crop_or_pad(images, crop_image_size[0], crop_image_size[1])

    # Stack them along the batch dim and return.
    concats = tf.stack(
        [crop_1, crop_2, crop_3, crop_4, crop_5],
        axis=0
    )
    labels = tf.tile(labels, multiples=[5])
    return concats, labels
