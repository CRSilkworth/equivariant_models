from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2

# from datasets import imagenet
import tensorflow.contrib.slim.nets
import image_net.labels.label_maps as lm
import squeeze_net.avoroshilov as av
import squeeze_net.model_def as md
import glob
import pprint
import os
import sys
import imp

def preprocess_image(image_file_name_placeholder, input_width, input_height, input_depth, input_mean):
    """Adds operations that perform JPEG decoding and resizing to the graph..
    Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.
    Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
    """
    image_string = tf.read_file(image_file_name_placeholder)
    decoded_image = tf.image.decode_jpeg(image_string, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    channels = tf.unstack(offset_image, axis=-1)
    img_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    return img_standardized
# def preprocess_image(image_file_name_placeholder, input_width, input_height, input_depth, input_mean):
#     """Adds operations that perform JPEG decoding and resizing to the graph..
#     Args:
#     input_width: Desired width of the image fed into the recognizer graph.
#     input_height: Desired width of the image fed into the recognizer graph.
#     input_depth: Desired channels of the image fed into the recognizer graph.
#     input_mean: Pixel value that should be zero in the image for the graph.
#     input_std: How much to divide the pixel values by before recognition.
#     Returns:
#     Tensors for the node to feed JPEG data into, and the output of the
#       preprocessing steps.
#     """
#     image_string = tf.read_file(image_file_name_placeholder)
#     decoded_image = tf.image.decode_jpeg(image_string, channels=input_depth)
#     decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
#     decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
#     resize_shape = tf.stack([input_height, input_width])
#     resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
#     resized_image = tf.image.resize_bilinear(decoded_image_4d,
#                                            resize_shape_as_int)
#     offset_image = tf.subtract(resized_image, input_mean)
#
#     return offset_image

def image_paths_and_labels(img_dir, img_label_file_name, label_def_file_name):
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

def are_close(a1, a2):
    return np.sum(np.abs(a1 - a2)) < 0.1
def main(cfg):

    # We need default size of image for a particular network.
    # The network was trained on images of that size -- so we
    # resize input image later in the code.

    with tf.Graph().as_default():

        image_file_name_placeholder = tf.placeholder(tf.string)
        is_training = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder(tf.float32)
        # image_tensor = tf.read_file(image_file_name_placeholder)
        # image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

        # Resize the input image, preserving the aspect ratio
        # and make a central crop of the resulted image.
        # The crop will be of the size of the default image size of
        # the network.

        processed_image = preprocess_image(
            image_file_name_placeholder,
            cfg.image_size[0],
            cfg.image_size[1],
            3,
            cfg.mean_rgb
        )

        # Networks accept images in batches.
        # The first dimension usually represents the batch size.
        # In our case the batch size is one.



        ###################################
        # NOTE: TAKE OUT
        # for key in sorted(loaded.keys()):
        #     if key.find('__') > -1:
        #         continue
        #     print "weights_dict['" + key + "'] = create_weights_biases(" + str(loaded[key][0].shape) + ")"
        # exit()
        ###################################
        loaded, _ = av.load_net(cfg.model_weights_path)
        net, end_points_1 = av.net_preloaded(loaded, processed_image, 'max', keep_prob=keep_prob, needs_classifier=True)
        # In order to get probabilities we apply softmax on the output.
        probabilities_1 = net['classifier_actv']

        logits, end_points_2 = md.squeeze_net_model(
            processed_image,
            weights_func=md.load_mat_weights,
            weights_func_kwargs={'data_path': cfg.model_weights_path},
            is_training=is_training
        )
        probabilities_2 = tf.nn.softmax(logits)

        with tf.Session() as sess:

            img_label_iter = image_paths_and_labels(
                cfg.img_dir,
                cfg.img_label_file_name,
                '../image_net/labels/synsets.txt'
                # cfg.label_def_file_name
            )
            right = 0
            wrong = 0
            for image_num, img_label in enumerate(img_label_iter):

                image_path, label_num, label_english = img_label
                # We want to get predictions, image as numpy matrix
                # and resized and cropped piece that is actually
                # being fed to the network.
                # probs_1, ends_1 = sess.run(
                #     [probabilities_1, end_points_1],
                #     feed_dict={image_file_name_placeholder: image_path, keep_prob: 1.}
                # )

                probs_2, ends_2 = sess.run(
                    [probabilities_2, end_points_2],
                    feed_dict={image_file_name_placeholder: image_path, is_training: False}
                )
                probs = probs_2

                sorted_inds = [i[0] for i in sorted(enumerate(-probs), key=lambda x:x[1])]

                # sorted_english = [lm.s_to_english[ind] for ind in sorted_inds[:5]]
                # print '----------------------'
                # print image_path, label_english
                # print sorted_english
                # print '----------------------'
                # print sorted_english, label_english

                # keys = ['conv1_w', 'conv1_b', 'conv', 'rel', 'x_1', 'x_2', 'x_3', 'x_5', 'x_9', 'x_10']
                # for key in keys:
                #     print key, are_close(ends_1[key], ends_2[key])
                # break
                if label_num in sorted_inds[:5]:
                    right += 1
                else:
                    wrong += 1

                if image_num % 10 == 0:
                    print float(right)/(right + wrong)
        # Show the downloaded image
        # plt.figure()
        # plt.imshow(np_image.astype(np.uint8))
        # plt.suptitle("Downloaded image", fontsize=14, fontweight='bold')
        # plt.axis('off')
        # plt.show()

        # Show the image that is actually being fed to the network
        # The image was resized while preserving aspect ratio and then
        # cropped. After that, the mean pixel value was subtracted from
        # each pixel of that crop. We normalize the image to be between [-1, 1]
        # to show the image.
        # plt.imshow( network_input / (network_input.max() - network_input.min()) )
        # plt.suptitle("Resized, Cropped and Mean-Centered input to network",
        #              fontsize=14, fontweight='bold')
        # plt.axis('off')
        # plt.show()

        # names = imagenet.create_readable_names_for_imagenet_labels()
        # for i in range(5):
        #     index = sorted_inds[i]
            # Now we print the top-5 predictions that the network gives us with
            # corresponding probabilities. Pay attention that the index with
            # class names is shifted by 1 -- this is because some networks
            # were trained on 1000 classes and others on 1001. VGG-16 was trained
            # on 1000 classes.
            # print('Probability %0.2f => [%s]' % (probabilities[index], names[index+1]))

        # res = slim.get_model_variables()
if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
