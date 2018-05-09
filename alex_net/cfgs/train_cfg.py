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

# Model parameters
model_type = md.AlexNet

# New model/continue training/pretrained model paramters
continue_training_run = '2018-04-24-08-11-38'
checkpoint_start_step = None
use_pretrained_weights = False

# Model definition parameters
# flip_constrain_fc6 = False
flip_constrain_fc6 = True
flip_weights_func = fc.flip_invariant_weights
# flip_weights_func = None
max_shape = False

# Input/output size parameters
image_size = [256, 256]
crop_image_size = [224, 224]
num_classes = 1000

# Random seed
random_seed = 3456
# Data input parameters
data_format = 'NCHW'
shuffle_buffer_size = 10000
num_dataset_threads = 20

# RGB parameters
rgb_distort = False
rgb_stddev = 0.1
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
summary_interval = 10000
checkpoint_interval = 10000
timeline_interval = 10000

# Training parameters
batch_size = 128
num_epochs = None
learning_rate = 0.000001
# weight_decay = 0.0005
weight_decay = 0.001
keep_prob = 0.5
momentum = 0.9

# Eval paramters
train_eval_max_iterations = 50000/128

# Checkpoint stuff
keep_last_n_checkpoints = 100

# Session configuration
gpu_memory_frac = 0.8
allow_soft_placement = False
log_device_placement = False
