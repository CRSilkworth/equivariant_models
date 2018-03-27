import glob

train_tfrecord_filepaths = sorted(glob.glob('/home/crsilkworth/tf_records_images/train-*'))
validation_tfrecord_filepaths = sorted(glob.glob('/home/crsilkworth/tf_records_images/validation-*'))

seed = 1

num_input_threads = 2
num_gpus = 1

num_classes = 1000
image_size = [256, 256]

keep_last_n_checkpoints = 10
summary_interval = 100
checkpoint_interval = 100
validation_interval = 100


batch_size = 128

# max_train_steps = 30001
num_epochs = 1
shuffle_buffer = 1000
learning_rate = 0.01
weight_decay = 0.0
batch_norm_decay = 0.9
