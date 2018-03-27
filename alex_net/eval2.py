from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2

# from datasets import imagenet
import tensorflow.contrib.slim.nets
import image_net.labels.label_maps as lm
import image_net.image_generator as ig
import alex_net.kratzert as kr
import alex_net.pjaehrling as pj
import glob
import pprint
import os
import sys
import imp


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
    img_standardized = tf.to_float(img_resize_crop)
    img_standardized = tf.subtract(img_standardized, imgnet_mean)

    # e.g. in my alexnet implementation the images are feed to the net in BGR format, NOT RGB
    channels = tf.unstack(img_standardized, axis=-1)
    img_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
    img_standardized = tf.expand_dims(img_standardized, 0)
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


def main(cfg):
    # We need default size of image for a particular network.
    # The network was trained on images of that size -- so we
    # resize input image later in the code.

    with tf.Graph().as_default():

        image_file_name_placeholder = tf.placeholder(tf.string)
        # image_tensor = tf.read_file(image_file_name_placeholder)
        # image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

        # Resize the input image, preserving the aspect ratio
        # and make a central crop of the resulted image.
        # The crop will be of the size of the default image size of
        # the network.
        processed_image = preprocess_image(
            image_file_name_placeholder,
            cfg.mean_rgb,
            cfg.image_size
        )

        # processed_image = preprocess_image(
        #     image_file_name_placeholder,
        #     cfg.image_size[0],
        #     cfg.image_size[1],
        #     3,
        #     cfg.mean_rgb
        # )

        # Networks accept images in batches.
        # The first dimension usually represents the batch size.
        # In our case the batch size is one.
        # processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure
        # the batch norm parameters. arg_scope is a very conveniet
        # feature of slim library -- you can define default
        # parameters for layers -- like stride, padding etc.
        keep_prob = tf.placeholder(tf.float32)

        # model = kr.AlexNet(processed_images, keep_prob, 1000, [], weights_path=cfg.model_weights_path)
        model = pj.AlexNet(processed_image)

        # In order to get probabilities we apply softmax on the output.
        # probabilities = tf.nn.softmax(model.fc8)
        probabilities = tf.nn.softmax(model.get_final_op())

        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        # init_fn = slim.assign_from_checkpoint_fn(
        #     cfg.model_weights_path,
        #     slim.get_model_variables('vgg_16')
        # )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Load weights
            model.load_initial_weights(sess)

            img_label_iter = image_paths_and_labels(
                cfg.img_dir,
                cfg.img_label_file_name
            )
            right = 0
            wrong = 0
            for image_num, img_label in enumerate(img_label_iter):

                image_path, label_num, label_english = img_label

                # We want to get predictions, image as numpy matrix
                # and resized and cropped piece that is actually
                # being fed to the network.
                network_input, probs = sess.run(
                    [processed_image, probabilities],
                    feed_dict={image_file_name_placeholder: image_path, keep_prob: 1.}
                )
                probs = probs[0]
                pred_labels = np.flip(np.argsort(probs), axis=0)[:5]

                sorted_english = [lm.s_to_english[l] for l in pred_labels]

                # print '----------------------'
                # print image_path, label_english
                # print sorted_english
                # print '----------------------'

                if label_num in pred_labels:
                    right += 1
                else:
                    wrong += 1

                if image_num % 100 == 0:
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
