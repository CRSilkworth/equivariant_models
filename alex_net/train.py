import numpy as np
import os
import tensorflow as tf
import image_net.data_augmentation as da
import alex_net.pjaehrling as pj
import alex_net.kratzert as kr
import alex_net.guerzhoy as gu
import pprint
import sys
import imp
import time
from tensorflow.python.client import timeline

def read_and_decode(filename_queue, batch_size, image_size=(256, 265), to_run=None):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/class/syn_code': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        }
    )
    to_run['filename'] = features['image/filename']
    to_run['syn_code'] = features['image/class/syn_code']
    to_run['text'] = features['image/class/text']

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.to_float(image)

    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label


def inputs(file_names, batch_size=128, image_size=(256, 256), num_epochs=None, crop_image_size=None, flip_axis=2, rgb_mean=[121.65, 116.67, 102.82], rgb_eigenvectors=None, rgb_eigenvalues=None, rgb_stddev=0.1, bgr=False, is_training=True, to_run=None):
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

            mean = tf.reshape(rgb_mean, [1, 1, 3])
            image = tf.subtract(image, mean)

            if is_training:
                image = crop_flip_distort(
                    image,
                    crop_image_size=crop_image_size,
                    flip_axis=flip_axis,
                    rgb_eigenvectors=rgb_eigenvectors,
                    rgb_eigenvalues=rgb_eigenvalues,
                    rgb_stddev=rgb_stddev,
                    to_run=to_run
                )
            else:
                image = crop_to_center(image, image_size, crop_image_size)
                image.set_shape(list(crop_image_size) + [3])

            if bgr:
                # Switch from rgb to bgr
                channels = tf.unstack(image, axis=-1)
                image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=3,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=1000
            )

            return images, labels


def crop_to_center(image, image_size, crop_image_size):
    crop_h = (image_size[0] - crop_image_size[0])/2
    crop_w = (image_size[1] - crop_image_size[1])/2

    image = image[crop_h: crop_h + crop_image_size[0], crop_w: crop_w + crop_image_size[1], :]
    return image


def crop_flip_distort(image, crop_image_size=None, flip_axis=2, rgb_eigenvectors=None, rgb_eigenvalues=None, rgb_stddev=0.1, to_run=None):
    with tf.name_scope('data_augmentation'):
        if crop_image_size is not None:
            image = tf.random_crop(image, [crop_image_size[0], crop_image_size[1], 3])
        if flip_axis is not None:
            image = tf.image.random_flip_left_right(image)
        if rgb_eigenvectors is not None:
            image = da.rgb_distortion(image, rgb_eigenvectors, rgb_eigenvalues, stddev=rgb_stddev, to_run=to_run)

    return image


def loss(logits, labels, num_classes=1000, weight_decay=0.0, to_run=None):
    with tf.name_scope('loss'):
        weights = [w for w in tf.all_variables() if not w.name.find('biases') > -1 and not w.name.find('Momentum') > -1]

        l2s = tf.stack([tf.nn.l2_loss(w) for w in weights])
        weight_norm = weight_decay * tf.reduce_sum(l2s)

        # oh_labels = tf.one_hot(labels, num_classes)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            # labels=oh_labels
            labels=labels
        )
        cross_entropy = tf.reduce_mean(cross_entropy)
        loss_val = cross_entropy + weight_norm
        to_run['weight_norm'] = weight_norm
        to_run['cross_entropy'] = cross_entropy
    return loss_val


def top_k_accuracy(logits, labels, batch_size, k=1):
    preds = tf.nn.softmax(logits)
    correct = tf.nn.in_top_k(predictions=preds, targets=labels, k=k)
    accuracy = tf.reduce_sum(tf.to_float(correct))/batch_size
    return accuracy


def optimize(loss, learning_rate=0.01, momentum=0.9, to_run=None):
    with tf.name_scope('optimize'):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum
        )

        return optimizer.minimize(loss=loss)


def main(cfg):
    with tf.Graph().as_default():
        to_run = {}

        # Input images and labels.
        images, labels = inputs(
            file_names=cfg.train_tfrecord_filepaths,
            batch_size=cfg.batch_size,
            num_epochs=cfg.num_epochs,
            crop_image_size=cfg.crop_image_size,
            flip_axis=cfg.flip_axis,
            rgb_mean=cfg.rgb_mean,
            rgb_eigenvectors=cfg.rgb_eigenvectors,
            rgb_eigenvalues=cfg.rgb_eigenvalues,
            rgb_stddev=cfg.rgb_stddev,
            bgr=cfg.bgr,
            is_training=cfg.is_training,
            to_run=to_run
        )

        if cfg.use_pretrained_weights:
            skip_layer = []
            retrain_layer = []
        else:
            skip_layer = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
            retrain_layer = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

        if cfg.model_type == 'pj':
            model = pj.AlexNet(
                images,
                num_classes=cfg.num_classes,
                keep_prob=cfg.keep_prob,
                retrain_layer=retrain_layer, weights_path=cfg.pretrained_weights_file
            )
            logits = model.get_final_op()
        elif cfg.model_type == 'kr':
            model = kr.AlexNet(
                images,
                keep_prob=cfg.keep_prob,
                num_classes=cfg.num_classes,
                skip_layer=skip_layer,
                weights_path=cfg.pretrained_weights_file
            )
            logits = model.fc8
        elif cfg.model_type == 'gu':
            model = gu.AlexNet(
                images,
                keep_prob=cfg.keep_prob,
                num_classes=cfg.num_classes,
                retrain_layer=skip_layer,
                weights_path=cfg.pretrained_weights_file
            )
            logits = model.get_logits()
        to_run['loss'] = loss(
            logits=logits,
            labels=labels,
            num_classes=cfg.num_classes,
            weight_decay=cfg.weight_decay,
            to_run=to_run
        )
        if not cfg.use_pretrained_weights:
            to_run['optimize'] = optimize(
                loss=to_run['loss'],
                learning_rate=cfg.learning_rate,
                momentum=cfg.momentum,
                to_run=to_run

            )
        to_run['accuracy'] = top_k_accuracy(
            logits=logits,
            labels=labels,
            batch_size=cfg.batch_size,
            k=5
        )
        pprint.pprint([w.name for w in tf.trainable_variables()])

        # The op for initializing the variables.
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        gpu_config = tf.GPUOptions(
            per_process_gpu_memory_fraction=cfg.gpu_memory_frac
        )
        config = tf.ConfigProto(
            allow_soft_placement=cfg.allow_soft_placement,
            log_device_placement=cfg.log_device_placement,
            gpu_options=gpu_config
        )
        with tf.Session(config=config) as sess:
            sess.run(init_op)

            if cfg.use_pretrained_weights and cfg.model_type in ('pj', 'kr', 'gu'):
                model.load_initial_weights(sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            try:
                begin_time = time.time()
                step = 0

                while not coord.should_stop():
                    start_time = time.time()

                    results = sess.run(
                        to_run,
                        options=options,
                        run_metadata=run_metadata
                        )

                    duration = time.time() - start_time
                    total_time = time.time() - begin_time

                    if step % cfg.print_every == 0 and step > 0:
                        print('------------------------------------------')
                        print('Total_time: %.3f') % (total_time,)
                        print('Step %d: (%.3f sec)' % (step, duration))
                        print('Time per step: %.3f' % (total_time/step,))
                        print ('Loss: %.3f' % results['loss'])
                        pprint.pprint(results)

                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open('timeline_step_%d.json' % (step/cfg.print_every), 'w') as f:
                            f.write(chrome_trace)
                    step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (cfg.num_epochs, step))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)


if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
