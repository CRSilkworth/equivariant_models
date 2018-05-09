# Created on May 31 2018
# Author: CRSilkworth
"""Colllection of functions which do miscellanous things. Mostly support for the training process."""
from tensorflow.python.client import timeline
import tensorflow as tf
import os
import pprint
import subprocess
import glob
import datetime

def dump_cfg_file(cfg_file_name, run_dir, step):
    """
    Dump the configuration file into the run directory so that all the parameters a run was trained using can be saved.
    Args:
        cfg_file_name: a string. The file name of the configuration file.
        run_dir: a string. The directory the training runs are stored in.
        step: an int. The step number the training process started from.
    """
    # Create the full cfg file path.
    new_cfg_file_name = cfg_file_name.split('/')[-1].replace('_cfg.py', '_' + str(step) + '_cfg.py')
    new_cfg_file_name = os.path.join(run_dir, new_cfg_file_name)

    # Copy the cfg to the run directory.
    command = "cp " + cfg_file_name + " " + new_cfg_file_name
    subprocess.call(command, shell=True)


def maybe_create_dir(*args):
    """
    Create a directory if it doesn't exist.

    Args:
        *args: strings. The path to the directory.
    Return:
        full_dir: a string. The directory that may have been created's name.
    """
    full_dir = os.path.join(*args)
    if not tf.gfile.Exists(full_dir):
        tf.gfile.MakeDirs(full_dir)
    return full_dir


def print_info(results, global_step, step_time, total_time, step=None):
    """
    Print some information about training.

    Args:
        results: a dictionary. Must have the loss as a key with it's value as the loss op.
        step: an int. the training step number.
        step_time: float (seconds). The length of time it took to run the step.
        total_time: float (seconds). The length of time the training has been running.
    """
    if step is None:
        step = global_step
    print('------------------------------------------')
    print('Global step %d: (%.3f sec)' % (global_step, step_time))
    print('Total_time: %.3f') % (total_time,)
    print('Time per step: %.3f' % (total_time/step,))
    print('Loss: %.3f' % results['loss'])
    print('Memory usage:')
    pprint.pprint(memory_usage())
    print('Results:')
    pprint.pprint(results)


def write_timeline(cur_run_dir, run_metadata, step):
    """
    Write a timeline json file to record the amount of various ops are taking.

    cur_run_dir: a string. The directory of the current run.
    run_metadata: The session metadata.
    step an int. The step number.
    """
    full_timeline_dir = maybe_create_dir(cur_run_dir, 'timelines')
    timeline_file_name = os.path.join(
        full_timeline_dir,
        'step_%d.json' % (step,)
    )

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()

    with open(timeline_file_name, 'w') as f:
        f.write(chrome_trace)


def write_checkpoint(saver, sess, cur_run_dir, global_step):
    """
    Write a checkpoint file, saving the model variables to disk.

    Args:
        saver: The checkpoint writer.
        sess: The training session.
        cur_run_dir: a string. The directory of the current run.
        global_step: The training step.
    """
    checkpoint_dir = maybe_create_dir(cur_run_dir, 'checkpoints')
    checkpoint_file_name = os.path.join(checkpoint_dir, 'model.ckpt')
    saver.save(sess, checkpoint_file_name, global_step)


def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = sizeof_fmt(int(parts[1]) * 1000)
    finally:
        if status is not None:
            status.close()
    return result


def sizeof_fmt(num, suffix='B'):
    """Convert the file size to human readable form."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def clip_checkpoints(run_dir, clip_step, time_str):
    checkpoint_dir = maybe_create_dir(run_dir, 'checkpoints')
    bk_checkpoint_dir = maybe_create_dir(run_dir, 'bk_checkpoints', time_str)

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model.ckpt-*.index'))

    for checkpoint_file in checkpoint_files:
        checkpoint_str = checkpoint_file[:-6]
        step_num = checkpoint_str.replace(checkpoint_dir, '').replace('model.ckpt-', '').replace('/', '')
        step_num = int(step_num)

        if not step_num > clip_step:
            continue

        subprocess.call('mv ' + checkpoint_str + '* ' + bk_checkpoint_dir, shell=True)


def clip_cfgs(run_dir, clip_step, time_str):
    cfg_dir = maybe_create_dir(run_dir)
    bk_cfg_dir = maybe_create_dir(run_dir, 'bk_cfgs', time_str)

    cfg_files = glob.glob(os.path.join(cfg_dir, 'train*cfg.py'))

    for cfg_file in cfg_files:
        step_num = cfg_file.split('_')[-2]
        step_num = int(step_num)

        if not step_num > clip_step:
            continue

        subprocess.call('mv ' + cfg_file + ' ' + bk_cfg_dir, shell=True)
