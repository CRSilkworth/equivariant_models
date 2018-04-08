# Created on May 31 2018
# Author: CRSilkworth
"""Contains the helper classes and functions for dealing with AlexNet data."""

import image_net.image_transformations as dt
import tensorflow as tf
import random


class AlexNetDataset:
    """
    Helper class which stores a tf.dataset object. It takes care of all the image transformations used in the AlexNet paper and handles the decoding, batching, and iterator initializers.

    """

    _additional_data = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/colorspace': tf.FixedLenFeature([], tf.string),
        'image/channels': tf.FixedLenFeature([], tf.int64),
        'image/class/syn_code': tf.FixedLenFeature([], tf.string),
        'image/class/text': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/filename': tf.FixedLenFeature([], tf.string)
    }

    def __init__(self, file_names, batch_size=128, num_epochs=None, max_iterations=None, shuffle_buffer_size=10000, image_size=(256, 265), rgb_mean=[0., 0., 0.], keep_prob=1., additional_data_keys=None, num_threads=None, to_run=None):
        shuffled_file_names = random.sample(file_names, len(file_names))
        self.dataset = tf.contrib.data.TFRecordDataset(shuffled_file_names)

        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.num_epochs = num_epochs
        self.shuffle_buffer_size = shuffle_buffer_size
        self.image_size = image_size
        self.rgb_mean = rgb_mean
        self.keep_prob = keep_prob
        self.num_threads = num_threads
        self.to_run = to_run

        self.additional_data_keys = additional_data_keys
        if additional_data_keys is None:
            self.additional_data_keys = []

        self.dataset = self.dataset.map(self._read_and_decode, num_parallel_calls=self.num_threads, output_buffer_size=self.batch_size)

        self.dataset = self.dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.max_iterations is None:
            self.dataset = self.dataset.repeat(num_epochs)

        else:
            self.dataset = self.dataset.take(max_iterations * batch_size)

        self.add_center_crop = False
        self.add_random_crop = False
        self.add_random_flip = False
        self.add_rgb_distort = False
        self.add_five_crops = False
        self.add_horizontal_flip = False

    def center_crop(self, crop_image_size):
        self.add_center_crop = True
        self.crop_image_size = crop_image_size

    def random_crop(self, crop_image_size):
        self.add_random_crop = True
        self.crop_image_size = crop_image_size

    def random_flip(self):
        self.add_random_flip = True

    def rgb_distort(self, rgb_eigenvectors, rgb_eigenvalues, stddev=0.1):
        self.add_rgb_distort = True
        self.rgb_eigenvectors = rgb_eigenvectors
        self.rgb_eigenvalues = rgb_eigenvalues
        self.rgb_stddev = stddev

    def five_crops(self, crop_image_size):
        self.add_five_crops = True
        self.crop_image_size = crop_image_size

    def horizontal_flip(self):
        self.add_horizontal_flip = True

    def _map_func(self, image, label):
        mean = tf.reshape(self.rgb_mean, [1, 1, 3])
        image = tf.subtract(image, mean)

        if self.add_center_crop:
            image = dt.crop_to_center(image, self.image_size, self.crop_image_size)
        if self.add_random_crop:
            image = tf.random_crop(image, [self.crop_image_size[0], self.crop_image_size[1], 3])
        if self.add_random_flip:
            image = tf.image.random_flip_left_right(image)
        if self.add_rgb_distort:
            image = dt.rgb_distortion(image, self.rgb_eigenvectors, self.rgb_eigenvalues, self.rgb_stddev)

        return image, label

    def add_five_crops_to_batch(self, image, label):
        return dt.add_five_crops_to_batch(image, label, self.image_size, self.crop_image_size)

    def add_horizontal_flip_to_batch(self, image, label):
        return dt.add_horizontal_flip_to_batch(image, label)

    def reinitializable_iterator(self, batched=True):
        dataset = self.dataset.map(self._map_func, num_parallel_calls=self.num_threads, output_buffer_size=self.batch_size)

        if self.add_five_crops or self.add_horizontal_flip:
            dataset = dataset.map(self._expand_dims)
        if self.add_five_crops:
            dataset = dataset.map(self.add_five_crops_to_batch)
        if self.add_horizontal_flip:
            dataset = dataset.map(self.add_horizontal_flip_to_batch)

        if batched:
            dataset = dataset.batch(self.batch_size)

        dataset = dataset.map(self._add_keep_prob)

        return tf.contrib.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes
        )

    def iterator_initializer(self, iterator, skip=None, batched=True):
        dataset = self.dataset.map(self._map_func, num_parallel_calls=self.num_threads, output_buffer_size=self.batch_size)

        if self.add_five_crops or self.add_horizontal_flip:
            dataset = dataset.map(self._expand_dims)
        if self.add_five_crops:
            dataset = dataset.map(self.add_five_crops_to_batch)
        if self.add_horizontal_flip:
            dataset = dataset.map(self.add_horizontal_flip_to_batch)

        if batched:
            dataset = dataset.batch(self.batch_size)

        if skip is not None:
            dataset = dataset.skip(skip)

        dataset = dataset.map(self._add_keep_prob)
        return iterator.make_initializer(dataset)

    def _read_and_decode(self, serialized_example):
        feature_dict = {
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        }
        for key in self.additional_data_keys:
            feature_dict[key] = AlexNetDataset._additional_data[key]

        features = tf.parse_single_example(
            serialized_example,
            features=feature_dict
        )

        for key in self.additional_data_keys:
            self.to_run[key] = features[key]

        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.to_float(image)
        image.set_shape(list(self.image_size) + [3])
        label = tf.cast(features['image/class/label'], tf.int32)

        return image, label

    def _subtract_mean(self, image, label):
        mean = tf.reshape(self.rgb_mean, [1, 1, 3])
        image = tf.subtract(image, mean)

        return image, label

    def _expand_dims(self, image, label):
        return tf.expand_dims(image, axis=0), tf.expand_dims(label, axis=0)

    def _add_keep_prob(self, images, labels):
        return images, labels, tf.constant(self.keep_prob)
