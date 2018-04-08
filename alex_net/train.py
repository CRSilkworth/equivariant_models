# Created on May 31 2018
# Author: CRSilkworth
"""
The training script for the AlexNet model. The script must be run with a configuration (cfg) file passed as an argument.

python train.py cfgs/train_cfg.py

The configuration file holds all the tuneable parameters for the model, training process and everthing else. The script outputs various files to the runs/<run_str> directory.

This script:
    * trains the AlexNet model.
    * prints information about the training process.
    * writes model checkpoint files and saves them to:
        runs/<run_str>/checkpoints/
    * writes summary files to be used with tensorboard:
        tensorboard --logdir runs/<run_str>/summaries/
    * writes timeline files to monitor op performance. To look at one
        * open chrome browser and go to page 'chrome://tracing'
        * click load
        * select a json file fromm runs/<run_str>/timelines/
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

    train_dataset - the dataset which is fed through the full pipeline up through the optimization set.

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
    train_dataset = dp.AlexNetDataset(
        file_names=cfg.train_tfrecord_filepaths,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_size=cfg.image_size,
        rgb_mean=cfg.rgb_mean,
        keep_prob=cfg.keep_prob,
        num_threads=cfg.num_dataset_threads,
        to_run=to_run
    )

    # Add random crop, random flip and pixel distortions to the images.
    train_dataset.random_crop(cfg.crop_image_size)
    if not cfg.flip_constrain_fc6:
        train_dataset.random_flip()
    train_dataset.rgb_distort(
        rgb_eigenvectors=cfg.rgb_eigenvectors,
        rgb_eigenvalues=cfg.rgb_eigenvalues,
        stddev=cfg.rgb_stddev
    )

    # Create the training eval dataset with info from the cfg file.
    train_eval_dataset = dp.AlexNetDataset(
        file_names=cfg.train_tfrecord_filepaths,
        batch_size=cfg.batch_size,
        max_iterations=cfg.train_eval_max_iterations,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_size=cfg.image_size,
        rgb_mean=cfg.rgb_mean,
        keep_prob=1.0,
        num_threads=cfg.num_dataset_threads,
        to_run=to_run
    )

    # Center crop the images
    train_eval_dataset.center_crop(cfg.crop_image_size)

    # Create the training dataset with info from the cfg file.
    val_eval_dataset = dp.AlexNetDataset(
        file_names=cfg.validation_tfrecord_filepaths,
        batch_size=cfg.batch_size,
        num_epochs=1,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_size=cfg.image_size,
        rgb_mean=cfg.rgb_mean,
        keep_prob=1.0,
        num_threads=cfg.num_dataset_threads,
        to_run=to_run
    )

    # Center crop the images
    val_eval_dataset.center_crop(cfg.crop_image_size)

    # Create the iterator shared by all three dataset
    data_iter = train_dataset.reinitializable_iterator()

    # Create the initializers of the datasets
    train_initializer = train_dataset.iterator_initializer(data_iter)
    train_eval_initializer = train_eval_dataset.iterator_initializer(data_iter)
    val_eval_initializer = val_eval_dataset.iterator_initializer(data_iter)

    return data_iter, train_dataset, train_initializer, train_eval_initializer, val_eval_initializer


def main(cfg, cfg_file_name=None):
    """
    The main training function.

    Args:
        cfg: the configuration module.
        cfg_file_name: the file name of the configuration module.

    """
    # If this is a new run create the current run directory, otherwise set the
    # current run directory to the one specified in the cfg.
    if cfg.continue_training_run is None:
        run_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        cur_run_dir = u.maybe_create_dir(cfg.run_dir, run_str)
    else:
        run_str = cfg.continue_training_run
        cur_run_dir = os.path.join(cfg.run_dir, run_str)

        # Copy last checkpoint to the restart checkpoint dir
        checkpoint_dir = u.maybe_create_dir(cur_run_dir, 'checkpoints')
        restart_dir = u.maybe_create_dir(cur_run_dir, 'restart_checkpoints')
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if cfg.checkpoint_start_step is None:
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        else:
            latest_checkpoint = os.path.join(
                checkpoint_dir,
                'model.ckpt-' + str(cfg.checkpoint_start_step)
            )

        print latest_checkpoint
        subprocess.call('cp ' + latest_checkpoint + ' ' + restart_dir, shell=True)

    with tf.Graph().as_default() as graph:

        # Define the dictionary which hold additional ops to be fed to sess.run
        to_run = {}

        # Define the global_step (aka the number of times optimize is called)
        global_step = tf.train.create_global_step()

        # Get all the dataset infromation
        data_iter, train_dataset, train_initializer, train_eval_initializer, val_eval_initializer = define_datasets(cfg, to_run)

        # Get the images, corresponding labels and keep probability from the
        # iterator
        images, labels, keep_prob = data_iter.get_next()

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

        # Calculate the loss
        loss = to.loss(
            logits=logits,
            labels=labels,
            num_classes=cfg.num_classes,
            weight_decay=cfg.weight_decay,
            to_run=to_run
        )
        to_run['loss'] = loss

        # Minimize the loss
        optimize = to.optimize(
            loss=loss,
            learning_rate=cfg.learning_rate,
            momentum=cfg.momentum,
            global_step=global_step,
            to_run=to_run

        )

        # Create the summary writers.
        summary_dir = u.maybe_create_dir(cur_run_dir, 'summaries')
        train_writer = tf.summary.FileWriter(
            os.path.join(summary_dir, 'train'),
            graph
        )
        val_writer = tf.summary.FileWriter(
            os.path.join(summary_dir, 'val'),
            graph
        )

        # Define the summary operations.
        summaries_dict = so.eval_summaries(
            loss=loss,
            logits=logits,
            labels=labels,
            to_run=to_run
        )
        var_summaries = so.variable_summaries()

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
            sess.run(train_initializer)

            # If this is a continuation of a previous run, load in the
            # variables from the checkpoint file. Or load in the weights from
            # http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
            if cfg.continue_training_run is not None:
                checkpoint_dir = os.path.join(cur_run_dir, 'checkpoints')
                saver.restore(
                    sess,
                    latest_checkpoint
                )
            elif cfg.use_pretrained_weights:
                to.load_weights(sess, cfg.pretrained_weights_file)

            gs = sess.run(global_step)
            print "STARTING STEP:", gs

            # Dump the configuration file into the run directory for
            # posteriority.
            if cfg_file_name is not None:
                u.dump_cfg_file(cfg_file_name, cur_run_dir, gs)

            begin_time = time.time()
            total_step_time = 0
            step = 0
            try:
                # Start the main train loop. Runs until user hits ctrl-c
                while True:
                    step_start_time = time.time()
                    results, _, gs = sess.run(
                        [to_run, optimize, global_step],
                        options=options,
                        run_metadata=run_metadata,
                        feed_dict={
                            keep_prob: cfg.keep_prob
                        }
                    )

                    # Keep track of times.
                    step_time = time.time() - step_start_time
                    total_step_time += step_time
                    total_time = time.time() - begin_time
                    step += 1

                    # Print info every interval.
                    if gs % cfg.print_interval == 0 and gs > 0:
                        u.print_info(results, step, step_time, total_time, step)

                    # Write a timeline every interval
                    if gs % cfg.timeline_interval == 0 and gs > 0:
                        u.write_timeline(cur_run_dir, run_metadata, step)

                    # Save the model every interval
                    if gs % cfg.checkpoint_interval == 0 and gs > 0:
                        u.write_checkpoint(saver, sess, cur_run_dir, global_step)

                    # Write summaries every interval.
                    if gs % cfg.summary_interval == 0 and gs > 0:
                        # write the variable summaries.
                        so.write_var_summaries(sess, train_writer, var_summaries, gs, total_step_time/step)

                        # Set data_iter to pull from train_eval and write the
                        # summaries.
                        sess.run(train_eval_initializer)
                        so.write_summaries(sess, train_writer, summaries_dict, gs)

                        # Set data_iter to pull from validation and write the
                        # summaries.
                        sess.run(val_eval_initializer)
                        so.write_summaries(sess, val_writer, summaries_dict, gs)

                        # Set the data_iter to pull from training data again.
                        train_initializer = train_dataset.iterator_initializer(data_iter)
                        sess.run(train_initializer)

            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (cfg.num_epochs, step))

            train_writer.close()
            val_writer.close()


if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg, abs_path)
