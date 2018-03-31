# Created on May 31 2018
# Author: CRSilkworth
"""Data augmenting and standard processing transformations performed on images."""
import tensorflow as tf


def crop_to_center(image, image_size, crop_image_size):
    """Take the center crop of an image of size image_size down to crop_size crop_image_size. If the the size of a dimension is odd then it takes the slightly shifted left or up."""
    crop_h = (image_size[0] - crop_image_size[0])/2
    crop_w = (image_size[1] - crop_image_size[1])/2

    image = image[crop_h: crop_h + crop_image_size[0], crop_w: crop_w + crop_image_size[1], :]
    return image


def rgb_distortion(image, rgb_eigenvectors, rgb_eigenvalues, stddev=0.1, to_run=None):
    """Linearly distort all the pixels of an image. The distortions are done in rgb space in the direction of the eigen vectors of the rgb covariance matrix. Each original pixel value [r_0, g_0, b_0] is distorted to:

    [r, g, b] = [r_0, g_0, b_0] +
        alpha_1 * lambda_1 * [p1_r, p1_g, p1_b] +
        alpha_2 * lambda_2 * [p2_r, p2_g, p2_b] +
        alpha_3 * lambda_3 * [p3_r, p3_g, p3_b]

    Where the alpha_i are random numbers drawn from a normal distribution of mean 0 and standard deviation of stddev. The lambda_i are the eigen values of the rgb covariance matrix and the pi's are the eigen vectors.
    """
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

    # Add the distortion and return
    return image + distortion


def add_horizontal_flip_to_batch(images, labels, to_run=None):
    # Flip along the width axis, stack with the original along the batch dim
    # and return.
    flipped = tf.reverse(images, axis=2)
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
