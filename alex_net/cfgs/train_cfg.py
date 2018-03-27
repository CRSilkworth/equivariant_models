import glob
import numpy as np
import os

# File paths
root_dir = '/home/crsilkworth/equivariant_models'
train_tfrecord_filepaths = sorted(glob.glob('/home/crsilkworth/tf_records_images/train-*'))
validation_tfrecord_filepaths = sorted(glob.glob('/home/crsilkworth/tf_records_images/validation-*'))
pretrained_weights_file = os.path.join(root_dir, 'alex_net/weights/bvlc_alexnet.npy')
checkpoint_dir = os.path.join(root_dir, 'alex_net/checkpoints')
summary_dir = os.path.join(root_dir, 'alex_net/summaries')
timeline_dir = os.path.join(root_dir, 'alex_net/timelines')

# Model parameters
is_training = True
use_pretrained_weights = False
model_type = 'gu'
crop_image_size = [224, 224]
keep_prob = 0.5
print_every = 1000
num_classes = 1000

# Random seed
seed = 1

# Data data_augmentation
flip_axis = 2
rgb_stddev = 0.3

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
keep_last_n_checkpoints = 10
summary_interval = 100
checkpoint_interval = 100
validation_interval = 100

# Training parameters
batch_size = 128
num_epochs = 90
shuffle_buffer = 1000
learning_rate = 0.01
weight_decay = 0.0005
momentum = 0.9

# Session configuration
gpu_memory_frac = 0.8
allow_soft_placement = False
log_device_placement = False


timeline_write_frequency = 100
