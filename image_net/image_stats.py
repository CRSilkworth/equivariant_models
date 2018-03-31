# Created on May 31 2018
# Author: CRSilkworth
"""
Script that extracts out various stats of the images fed to it. The script must be run with a configuration (cfg) file passed as an argument.

python image_stats.py cfgs/image_stats.py

The script calculates:
    * The mean rgb values of every pixel of every image.
    * The covariance matrix between the red green and blue pixels of every pixel of every image.
    * The eigen vectors and eigen values of the covariance matrix.
"""
import numpy as np
import os
import tensorflow as tf
import sys
import imp
import time
import pprint


def rgb_mean(images):
    """
    Calculates the mean of the rgb vector across every pixel, across every image.

    Args:
        images: A float tensor of shape [batch_size, image_height, image_width, 3]
    Returns:
        rgb_mean: A tensor of shape [3] with the mean red, green and blue values respectively.
    """
    return tf.reduce_mean(images, axis=[0, 1, 2])


def rgb_covariance(images, batch_size, image_size):
    """
    Calculates the covariance of the red, green and blue pixel values across every pixel, across every image.

    Args:
        images: A tensor of shape [batch_size, image_height, image_width, 3]
        batch_size: an int. The number of images in the batch.
        image_size: a pair of ints. The height and width of each image.
    Returns:
        rgb_covariance: A tensor of shape [3, 3]. The covariance between the red, green and blue values.
    """
    n = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    reshaped = tf.reshape(images, [height*width*n, 3])
    batch_covariance = tf.matmul(tf.transpose(reshaped), reshaped)

    return batch_covariance/tf.to_float(n * height * width)


def read_and_decode(filename_queue, batch_size, image_size=(256, 265), to_run=None):
    """
    Read and decode an image label pair from a tfrecord file.

    Args:
        filename_queue: a string input producer object with all the tfrecord file name images.
        batch_size: an int. The number of images in the batch.
        image_size: a pair of ints. The height and width of each image.
        to_run: a dictionary which can be passed to pick up any ops inside the function that need to be fed to sess.run. Useful for debugging.
    Returns:
        image: a uint8 tensor of shape [image_height, image_width, 3].
        label: an int32 scalar tensor.

    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        }
    )

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image.set_shape(list(image_size) + [3])
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label


def inputs(file_names, batch_size=128, image_size=(256, 256), num_epochs=None, mean_rgb=[121.65, 116.67, 102.82], to_run=None):
    """
    Read input data num_epochs times.

    Args:
        file_names: a list of strings. the file names of the tfrecords.
        batch_size: an int. The number of images in the batch.
        image_size: a pair of ints. The height and width of each image.
        mean_rgb: A tensor of shape [3] with the mean red, green and blue values respectively.
        num_epochs: The number of times to run through the dataset.
        to_run: a dictionary which can be passed to pick up any ops inside the function that need to be fed to sess.run. Useful for debugging.
    Returns:
        images: A float tensor of shape [batch_size, image_height, image_width, 3]
        label: an int32 tensor of shape [batch_size]. The labels of the images.

    """
    with tf.name_scope('input'):
        with tf.device('/cpu:0'):
            filename_queue = tf.train.string_input_producer(
                file_names, num_epochs=num_epochs)

            image, label = read_and_decode(filename_queue, batch_size, image_size, to_run)

            mean = tf.reshape(mean_rgb, [1, 1, 3])
            image = tf.to_float(image)
            image = tf.subtract(image, mean)

            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=3,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000
            )

            return images, labels

def main(cfg):
    """The main image stats function."""
    with tf.Graph().as_default():
        # Define the dictionary which hold additional ops to be fed to sess.run
        to_run = {}

        # Get the images and labels tensors from the tfrecord files.
        images, labels = inputs(
            file_names=cfg.train_tfrecord_filepaths,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_epochs=cfg.num_epochs,
            to_run=to_run
        )

        # Define the session
        sess = tf.Session()

        # Get the batch mean and batch rgb covariance of the images.
        to_run['batch_mean'] = rgb_mean(images)
        to_run['batch_covariance'] = rgb_covariance(images, cfg.batch_size, cfg.image_size)

        # Initialize the variables
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)

        # Define the coordinator and start the que runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            begin_time = time.time()
            step = 0

            # Iterate through the data.
            while not coord.should_stop():
                start_time = time.time()

                # Run everything in the to_run dictionary.
                results = sess.run(to_run)

                # Calculate the full mean and covariance from the batches.
                if step == 0:
                    # If this is the first step just set them to batch values.
                    mean = results['batch_mean']
                    covariance = results['batch_covariance']
                else:
                    # Otherwise scale previous mean value by (n - 1)/n and add
                    # to batch values scaled by 1/n/
                    a = (step - 1.)/step
                    b = 1./step
                    mean = a * mean + b * results['batch_mean']
                    covariance = a * covariance + b * results['batch_covariance']

                # Get the times.
                duration = time.time() - start_time
                total_time = time.time() - begin_time

                # Print info every 100 steps.
                if step % 100 == 0 and step > 0:
                    print('------------------------------------------')
                    pprint.pprint(results)
                    print('Step %d: (%.3f sec)' % (step, duration))
                    print('Time per step: %.3f' % (total_time/step,))
                    print()
                    print('RGB MEAN:')
                    print(mean)
                    print()
                    print('RGB COVARIANCE:')
                    print(covariance)

                    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                    print()
                    print('RGB COVARIANCE EIGENVALUES:')
                    print(eigenvalues)
                    print()
                    print('RGB COVARIANCE EIGENVECTORS:')
                    print(eigenvectors)
                    print('------------------------------------------')
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (cfg.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
