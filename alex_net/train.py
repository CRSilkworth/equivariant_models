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
import image_net.pipeline as inputs
import glob
import pprint
import os
import sys
import imp


def preprocess_image(images, mean_rgb, rgb_eigenvectors, rgb_eigenvalues, crop_image_size=(224, 224), is_training):
    batch_size = tf.shape(images)[0]
    # Subtract the imagenet mean (mean over all imagenet images)
    imagenet_mean = tf.reshape(mean_rgb, [1, 1, 3])
    imagenet_mean = tf.expand_dims(imagenet_mean, 0)
    imagenet_mean = tf.tile(imagenet_mean, multiples=[batch_size, 1, 1, 1])
    images_standardized = tf.subtract(images, imagenet_mean)

    if is_training:
        images_standardized = rgb_distortion(images_standardized, rgb_eigenvectors, rgb_eigenvalue)
        images_standardized = five_crops(images_standardized, crop_image_size)
        images_standardized = horizontal_flip(images_standardized)
    else:
        images_standardized = tf.image.resize_image_with_crop_or_pad(images, crop_image_size[0], crop_image_size[1])

    # e.g. in my alexnet implementation the images are feed to the net in BGR format, NOT RGB
    channels = tf.unstack(images_standardized, axis=-1)
    images_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    images_standardized = tf.expand_dims(images_standardized, 0)
    return img_standardized


def rgb_distortion(images, rgb_eigenvectors, rgb_eigenvalues):
    height = images.shape[1]
    width = images.shape[2]

    rgb_distortion = []
    for color_num in xrange(3):
        # Pull out the color eigenvector and tile by the number of images in the
        # batch.
        eigenvector = rgb_eigenvectors[color_num]
        eigenvector = tf.expand_dims(eigenvector, 0)
        eigenvector = tf.tile(eigenvector, multiples=[batch_size, 1])

        # Pull out the color eigenvector and tile by the number of images in the
        # batch and 3 times to match the eigenvector dims
        eigenvalue = rgb_eigenvalue[color_num]
        eigenvalue = tf.expand_dims(tf.expand_dims(eigenvalue, 0), 0)
        eigenvalue = tf.tile(eigenvalue, multiples=[batch_size, 3])

        # Get the random number drawn from a normal and tile it to match the
        # shape of eigenvector tensor
        random = tf.random_normal([batch_size], stddev=0.1)
        random = tf.expand_dims(random, -1)
        random = tf.tile(random, multiples=[1, 3])

        # Multiply them together and sum along the eigenvector's original dim.
        rgb_distortion.append(
            tf.reduce_sum(random * eigenvalue * eigenvector, axis=-1)
        )
    # Stack them into a single tensor, and tile to match the images dims.
    rgb_distortion = tf.stack(rgb_distortion, axis=-1)
    rgb_distortion = tf.expand_dims(tf.expand_dims(rgb_distortion, 1))
    rgb_distortion = tf.tile(rgb_distortion, multiples=[1, height, width, 1])

    # Add the distortion and return
    return images + rgb_distortion


def five_crops(images, crop_image_size):
    # Four corner crops
    crop_1 = images[:, :crop_image_size[0], :crop_image_size[1], :]
    crop_2 = images[:, -crop_image_size[0]:, :crop_image_size[1], :]
    crop_3 = images[:, :crop_image_size[0], :-crop_image_size[1], :]
    crop_4 = images[:, -crop_image_size[0]:, :-crop_image_size[1], :]

    # Center crop
    crop_5 = tf.image.resize_image_with_crop_or_pad(images, crop_image_size[0], crop_image_size[1])

    # Stack them along the batch dim and return.
    concats = tf.stack(
        [crop_1, crop_2, crop_3, crop_4, crop_5],
        axis=0
    )
    return concats


def horizontal_flip(images):
    # Flip along the width axis, stack with the original along the batch dim
    # and return.
    flipped = tf.reverse(images, axis=2)
    return tf.stack([images, flipped], axis=0)


def main(cfg):
    # We need default size of image for a particular network.
    # The network was trained on images of that size -- so we
    # resize input image later in the code.

    with tf.Graph().as_default():

        pipeline = inputs.Pipeline(cfg, sess)
        examples, labels = pipeline.data
        images = examples['image']

        processed_image = preprocess_image(
            images,
            cfg.mean_rgb,
            cfg.rgb_eigenvectors,
            cfg.rgb_eigenvalues,
            cfg.crop_image_size
            is_training=True
        )

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
