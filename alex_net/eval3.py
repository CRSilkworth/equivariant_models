################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import image_net.labels.label_maps as lm

import glob
import tensorflow as tf

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]
def preprocess_image(image_file_name_placeholder, mean_rgb, image_size=(227, 227)):
    image_string = tf.read_file(image_file_name_placeholder)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # Get the input Dimensions
    input_shape = tf.shape(image_decoded)
    input_height = tf.to_float(input_shape[0])
    input_width = tf.to_float(input_shape[1])

    # Find out which side has the smallest scalling factor, so we resize by this
    scale_height = tf.to_float(image_size[0]) / input_height
    scale_width = tf.to_float(image_size[1]) / input_width
    scale = tf.cond(
        tf.greater(scale_height, scale_width),
        lambda: scale_height,
        lambda: scale_width
    )

    new_height = tf.to_int32(input_height * scale)
    new_width = tf.to_int32(input_width * scale)

    # Resize (keep ratio) and Crop to fit output dimensions
    img_resize = tf.image.resize_images(image_decoded, [new_height, new_width])
    img_resize_crop = tf.image.resize_image_with_crop_or_pad(img_resize, image_size[0], image_size[1])

    # Subtract the imagenet mean (mean over all imagenet images)
    imgnet_mean = tf.reshape(mean_rgb, [1, 1, 3])
    img_float = tf.to_float(img_resize_crop)
    img_standardized = tf.subtract(img_float, imgnet_mean)

    # e.g. in my alexnet implementation the images are feed to the net in BGR format, NOT RGB
    channels = tf.unstack(img_standardized, axis=-1)
    img_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    return img_standardized

def image_paths_and_labels(img_dir, img_label_file_name):
    # Pull out the labels from the label file.
    if img_label_file_name is not None:
        with open(img_label_file_name, 'r') as img_label_file:
            img_labels = [int(l_num) for l_num in img_label_file]

    # Gather all the image paths. It will be assumed their sorted names
    # correspond to the same order as the labels in img_label_file
    img_paths = glob.glob(os.path.join(img_dir, '*.JPEG'))
    img_paths.sort()

    # Create the full list of image information.
    for line_num, img_path in enumerate(img_paths):
        label_num = None
        label_english = None
        if img_label_file_name is not None:
            label_num = lm.l_to_s[img_labels[line_num]]
            label_english = lm.s_to_english[label_num]
        yield img_path, label_num, label_english


################################################################################
#Read Image, and change to BGR


# im1 = (imread("laska.png")[:,:,:3]).astype(float32)
# im1 = im1 - mean(im1)
# im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
#
# im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
# im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
# net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
net_data = load("weights/bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



# x = tf.placeholder(tf.float32, (None,) + xdim)
image_file_name_placeholder = tf.placeholder(tf.string)
x = preprocess_image(image_file_name_placeholder, np.array([123.68, 116.78, 103.94], dtype=np.float32), image_size=(227, 227))
x = tf.expand_dims(x, 0)

#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


right = 0
wrong = 0
for image_num, (img_path, label_num, label_english) in enumerate(image_paths_and_labels('../image_net/images/val/', '../image_net/labels/val_labels.txt')):
    t = time.time()

    # im1 = (imread(img_path)[:,:,:3]).astype(float32)
    # im1 = im1 - mean(im1)
    # im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

    probs = sess.run(prob, feed_dict = {image_file_name_placeholder:img_path})

    probs = probs[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-probs), key=lambda x:x[1])]

    sorted_english = [lm.s_to_english[ind] for ind in sorted_inds[:5]]
    # print '----------------------'
    # print image_path, label_english
    # print sorted_english
    # print '----------------------'

    if label_num in sorted_inds[:5]:
        right += 1
    else:
        wrong += 1

    if image_num % 100 == 0:
        print float(right)/(right + wrong)
################################################################################

#Output:
