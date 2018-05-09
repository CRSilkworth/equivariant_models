# Created on May 31 2018
# Author: CRSilkworth
"""
The training script for the AlexNet model. The script must be run with a configuration (cfg) file passed as an argument.

python train.py cfgs/train_cfg.py

The configuration file holds all the tuneable parameters for the model, training process and everthing else. The script outputs various files to the runs/<run_str> directory.

"""
import alex_net.data_pipeline as dp
import alex_net.train_ops as to
import alex_net.utils as u
import alex_net.summary_ops as so
import os
import tensorflow as tf
import sys
import imp
import time
import datetime
import subprocess
import pprint
import numpy as np


def configs(cfg):
    """Helper function which holds all the session configuration info."""
    # Define any gpzu options.
    gpu_config = tf.GPUOptions(
        per_process_gpu_memory_fraction=cfg.gpu_memory_frac
    )
    # Define session configuration.
    config = tf.ConfigProto(
        allow_soft_placement=cfg.allow_soft_placement,
        log_device_placement=cfg.log_device_placement,
        gpu_options=gpu_config
    )

    # Define the session options and metadata for timelines.
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    return config, options, run_metadata

def define_datasets(cfg, to_run=None):
    """
    Define the three helper datasets which are used throughout training.

    train_eval - the dataset which is used build summaries of performance metrics on the train set.

    val_eval - the dataset which is used build summaries of performance metrics on the valiation set.

    Args:
        cfg: the configuration module.
        to_run: a dictionary which can be passed to pick up any ops inside the function that need to be fed to sess.run. Useful for debugging.
    Returns:
        data_iter: The iterator shared by the three dataset which is passed through the graph.
        train_dataset: The AlexNetDataset object used for training.
        train_initializer: The initializer fed to sess.run to tell data_iter to start pulling data from the start of the train_dataset.
        train_eval_initializer: The initializer fed to sess.run to tell data_iter to start pulling data from the start of the train_eval_dataset.
        val_eval_initializer: The initializer fed to sess.run to tell data_iter to start pulling data from the start of the val_eval_dataset.
    """

    # Create the training dataset with info from the cfg file.
    val_eval_dataset = dp.AlexNetDataset(
        file_names=cfg.validation_tfrecord_filepaths,
        batch_size=1,
        num_epochs=1,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_size=cfg.image_size,
        rgb_mean=cfg.rgb_mean,
        keep_prob=1.0,
        num_threads=cfg.num_dataset_threads,
        to_run=to_run
    )

    val_eval_dataset.five_crops(cfg.crop_image_size)

    if not cfg.flip_constrain_fc6:
        val_eval_dataset.horizontal_flip()

    # Create the iterator shared by all three dataset
    data_iter = val_eval_dataset.reinitializable_iterator(batched=False)

    # Create the initializers of the datasets
    val_eval_initializer = val_eval_dataset.iterator_initializer(data_iter, batched=False)

    return data_iter, val_eval_dataset, val_eval_initializer


def main(cfg, cfg_file_name=None):
    """
    The main training function.

    Args:
        cfg: the configuration module.
        cfg_file_name: the file name of the configuration module.

    """
    # Set the current run directory to the one specified in the cfg.
    run_str = cfg.continue_training_run
    cur_run_dir = os.path.join(cfg.run_dir, run_str)

    with tf.Graph().as_default() as graph:

        # Define the dictionary which hold additional ops to be fed to sess.run
        to_run = {}

        # Get all the dataset infromation
        data_iter, val_eval_dataset, val_eval_initializer = define_datasets(cfg, to_run)

        # Get the images, corresponding labels and keep probability from the
        # iterator
        images, labels, keep_prob = data_iter.get_next()
        print labels
        # Define the model
        model = cfg.model_type(
            images,
            num_classes=cfg.num_classes,
            keep_prob=keep_prob,
            data_format=cfg.data_format,
            flip_constrain_fc6=cfg.flip_constrain_fc6,
            flip_weights_func=cfg.flip_weights_func
        )

        # Get the logits produced by the model
        logits = model.get_logits()

        av_logits = tf.reduce_mean(logits, axis=0)
        av_logits = tf.expand_dims(av_logits, axis=0)

        labels = labels[:1]
        # Calculate the loss
        loss = to.loss(
            logits=av_logits,
            labels=labels,
            num_classes=cfg.num_classes,
            weight_decay=cfg.weight_decay,
            to_run=to_run
        )
        to_run['loss'] = loss

        # Define the summary operations.
        summaries_dict = so.eval_summaries(
            loss=loss,
            logits=av_logits,
            labels=labels,
            to_run=to_run
        )

        # Create the saver to save the model at checkpoints.
        saver = tf.train.Saver(max_to_keep=cfg.keep_last_n_checkpoints)

        # Define the variable initializer operation
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        # Get the session configuration and start the session.
        sess_config, options, run_metadata = configs(cfg)
        with tf.Session(config=sess_config, graph=graph) as sess:

            # Initialize the variables
            sess.run(init_op)

            # Set the data_iter to pull from the train_dataset
            sess.run(val_eval_initializer)

            # If this is a continuation of a previous run, load in the
            # variables from the checkpoint file. Or load in the weights from
            # http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
            if cfg.continue_training_run is not None:
                checkpoint_dir = os.path.join(cur_run_dir, 'checkpoints')
                if cfg.checkpoint_start_step is None:
                    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                else:
                    latest_checkpoint = os.path.join(
                        checkpoint_dir,
                        'model.ckpt-' + str(cfg.checkpoint_start_step)
                    )
                saver.restore(
                    sess,
                    latest_checkpoint
                )
            elif cfg.use_pretrained_weights:
                to.load_weights(sess, cfg.pretrained_weights_file)

            begin_time = time.time()
            total_step_time = 0
            all_summaries = {}
            step = 0
            try:
                while True:
                    step_start_time = time.time()

                    results, summaries = sess.run(
                        [to_run, summaries_dict],
                        options=options,
                        run_metadata=run_metadata,
                        # feed_dict={
                        #     keep_prob: 1.
                        # }
                    )
                    for key in summaries:
                        all_summaries.setdefault(key, [])
                        all_summaries[key].append(summaries[key])

                    # Keep track of times.
                    step_time = time.time() - step_start_time
                    total_step_time += step_time
                    total_time = time.time() - begin_time

                    all_summaries.setdefault('step_time', [])
                    all_summaries['step_time'].append(step_time)
                    step += 1

                    if step % cfg.print_interval == 0 and step > 0:
                        u.print_info(results, step, step_time, total_time)
            except tf.errors.OutOfRangeError:
                pass

            for key in all_summaries:
                all_summaries[key] = np.mean(all_summaries[key])

            pprint.pprint(all_summaries)


if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg, abs_path)
