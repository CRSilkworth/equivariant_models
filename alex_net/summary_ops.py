# Created on May 31 2018
# Author: CRSilkworth
"""Operations and functions used in the process of writing summaries for the training process."""
import alex_net.train_ops as to
import alex_net.utils as u
import tensorflow as tf
import numpy as np
import datetime
import subprocess
import glob
import os


def eval_summaries(loss, logits, labels, to_run=None):
    """
    Evaulation summary operations. The operations which will be saved to summary files.

    Args:
        loss: a scalar tensor of he cross entropy + weight norm
        logits: The logits operation outputted by the model.
        labels: The image labels.
        to_run: a dictionary which can be passed to pick up any ops inside the function that need to be fed to sess.run. Useful for debugging.
    Returns:
        summaries_dict: A dictionary of all the summary operations.
    """
    summaries_dict = {
        'loss': loss,
        'cross_entropy': to_run['cross_entropy'],
        'weight_norm': to_run['weight_norm'],
        'top_1_accuracy': to.top_k_accuracy(
            logits=logits,
            labels=labels,
            k=1
        ),
        'top_5_accuracy': to.top_k_accuracy(
            logits=logits,
            labels=labels,
            k=5
        )

    }

    return summaries_dict


def variable_summaries():
    """Get all the variable distribution histograms and put them into a summary."""
    var_summaries = []
    for var in tf.trainable_variables():
        var_summaries.append(tf.summary.histogram(var.op.name, var))
    var_summaries = tf.summary.merge(var_summaries)
    return var_summaries


def write_var_summaries(sess, writer, var_summaries, step, time_per_step=None, run_metadata=None):
    """
    Write the variable summaries as well as the training step time

    Args:
        sess: The training session.
        writer: The summary writer
        var_summaries: a summary operation. The variables summaries.
        step: an int. the training step number.
        time_per_step: a float (seconds) The length of time a training step takes.
    """
    # If a time_per_step is given then create a summary and add it.
    if time_per_step is not None:
        summary = tf.Summary()
        summary.value.add(tag='time_per_step', simple_value=np.mean(time_per_step))
        writer.add_summary(summary, step)

    # Add metadata
    if run_metadata is not None:
        writer.add_run_metadata(run_metadata, 'step%d' % step)

    # Add the summary to the writer and flush.
    writer.add_summary(sess.run(var_summaries), step)
    writer.flush()


def write_summaries(sess, writer, summaries_dict, step):
    """
    Write the evaluation summary.

    Args:
        sess: The training session.
        writer: The summary writer.
        summaries_dict: A dictionary of all the summary operations.
        step: an int. the training step number.
        time_per_step: a float (seconds) The length of time a training step takes.
    """
    results_dict = {}
    try:
        # Go through the entire dataset.
        while True:
            # Run the summary operations.
            results = sess.run(summaries_dict)

            # Add each summary operation to its respective list.
            for key in results:
                results_dict.setdefault(key, [])
                results_dict[key].append(results[key])

    except tf.errors.OutOfRangeError:
        pass

    summary = tf.Summary()
    for key in results_dict:
        # Take the average of all the steps and add it to a summary.
        mean = np.mean(results_dict[key])
        summary.value.add(tag=key, simple_value=mean)

    # Add the summary to the writer and flush.
    writer.add_summary(summary, step)
    writer.flush()


def clip_summaries(run_dir, clip_step, time_str):
    summary_dir = u.maybe_create_dir(run_dir, 'summaries')
    bk_summary_dir = u.maybe_create_dir(run_dir, 'bk_summaries', time_str)

    subprocess.call('mv ' + summary_dir + '/* ' + bk_summary_dir, shell=True)

    for dir in ['train', 'val']:
        writer = tf.summary.FileWriter(os.path.join(summary_dir, dir))

        summary_files = sorted(glob.glob(os.path.join(bk_summary_dir, dir, '*')))
        for summary_file in summary_files:
            for e in tf.train.summary_iterator(summary_file):
                if e.step > clip_step:
                    continue

                writer.add_event(e)
        writer.flush()
        writer.close()
