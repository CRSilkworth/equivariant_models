import numpy as np

model_weights_path = 'weights/sqz_full.mat'
img_dir = '../image_net/images/val/'
img_label_file_name = '../image_net/labels/val_labels.txt'
label_def_file_name = '../image_net/labels/label_defs.txt'
allowed_extensions = ['JPEG']
image_size = (227, 227)
mean_rgb = np.array([123.68, 116.78, 103.94], dtype=np.float32)  # RGB
