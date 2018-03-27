import numpy as np
import os
import tensorflow as tf
import sys
import imp
import time
import pprint


def rgb_mean(images):
    return tf.reduce_mean(images, axis=[0, 1, 2])


def rgb_covariance(images, batch_size, image_size):
    n = tf.shape(images)[0]
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    reshaped = tf.reshape(images, [height*width*n, 3])
    batch_covariance = tf.matmul(tf.transpose(reshaped), reshaped)

    return batch_covariance/tf.to_float(n * height * width)


def read_and_decode(filename_queue, batch_size, image_size=(256, 265), to_run=None):
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
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape(list(image_size) + [3])
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label


def inputs(file_names, batch_size=128, image_size=(256, 256), num_epochs=None, mean_rgb=[121.65, 116.67, 102.82], to_run=None):
    """
    Read input data num_epochs times.

    Args:
    data_dir: The directory with the tfrecord files.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or None to
       train forever.
    Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
    """
    with tf.name_scope('input'):
        with tf.device('/cpu:0'):
            filename_queue = tf.train.string_input_producer(
                file_names, num_epochs=num_epochs)

            image, label = read_and_decode(filename_queue, batch_size, image_size, to_run)

            # mean = tf.reshape(mean_rgb, [1, 1, 3])/255.
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
    with tf.Graph().as_default():
        to_run = {}
        # Input images and labels.
        images, labels = inputs(
            file_names=cfg.train_tfrecord_filepaths,
            batch_size=cfg.batch_size,
            image_size=cfg.image_size,
            num_epochs=cfg.num_epochs,
            to_run=to_run
        )
        to_run['global_step'] = tf.Variable(1)
        sess = tf.Session()

        to_run['batch_mean'] = rgb_mean(images)
        to_run['batch_covariance'] = rgb_covariance(images, cfg.batch_size, cfg.image_size)

        # The op for initializing the variables.
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)
        tf.train.global_step(sess, to_run['global_step'])

        # Create a session for running operations in the Graph.
        # Initialize the variables (the trained variables and the
        # epoch counter).

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            begin_time = time.time()
            step = 0

            while not coord.should_stop():
                start_time = time.time()

                results = sess.run(to_run)
                if step == 0:
                    mean = results['batch_mean']
                    covariance = results['batch_covariance']
                else:
                    a = (step - 1.)/step
                    b = 1./step
                    mean = a * mean + b * results['batch_mean']
                    covariance = a * covariance + b * results['batch_covariance']
                duration = time.time() - start_time
                total_time = time.time() - begin_time
                # Print an overview fairly often.

                if step % 10 == 0 and step > 0:
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
