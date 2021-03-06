import glob
import numpy as np
import os
import alex_net.pjaehrling as pj
import alex_net.kratzert as kr
import alex_net.guerzhoy as gu
import alex_net.model_def as md
import constrained_weights.flip_constrained as fc

# File paths
root_dir = '/home/crsilkworth/equivariant_models'
run_dir = os.path.join(root_dir, 'alex_net/runs')
train_tfrecord_filepaths = sorted(glob.glob('/home/crsilkworth/tf_records_images/train-*'))
validation_tfrecord_filepaths = sorted(glob.glob('/home/crsilkworth/tf_records_images/validation-*'))
pretrained_weights_file = os.path.join(root_dir, 'alex_net/weights/bvlc_alexnet.npy')
checkpoint_dir = os.path.join(root_dir, 'alex_net/checkpoints')
summary_dir = os.path.join(root_dir, 'alex_net/summaries')
timeline_dir = os.path.join(root_dir, 'alex_net/timelines')

# Model parameters
model_type = md.AlexNet

# Setup parameters
continue_training_run = '2018-04-24-08-11-38'
checkpoint_start_step = 550000
flip_constrain_fc6 = True
flip_weights_func = fc.flip_invariant_weights
# flip_weights_func = None
max_shape = False
use_pretrained_weights = False
image_size = [256, 256]
crop_image_size = [224, 224]
keep_prob = 0.5
num_classes = 1000
num_dataset_threads = 20

# Data input parameters
data_format = 'NCHW'
shuffle_buffer_size = 10000

# Data data_augmentation
rgb_stddev = 0.1

# RGB summary attributes
rgb_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
rgb_eigenvectors = np.array(
    [
        [4.0456429e-01, -7.1764714e-01, -5.6684244e-01],
        [-8.1423664e-01, -4.7433414e-04, -5.8053291e-01],
        [4.1634887e-01, 6.9640672e-01, -5.8452654e-01],
    ],
    dtype=np.float32
)
rgb_eigenvalues = np.array(
    [258.35184, 1167.8958, 13727.228],
    dtype=np.float32
)
bgr = True

# Intervals
print_interval = 1000
timeline_interval = 10000

# Training parameters
batch_size = 128
weight_decay = 0.0005


# Eval paramters
train_eval_max_iterations = 50000/128

# Checkpoint stuff
keep_last_n_checkpoints = 10

# Session configuration
gpu_memory_frac = 0.8
allow_soft_placement = False
log_device_placement = False
