from matplotlib import pyplot as plt

import numpy as np
import os
import tensorflow as tf
import urllib2

# from datasets import imagenet
import tensorflow.contrib.slim.nets
import image_net.labels.label_maps as lm
import image_net.image_generator as ig
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
#     channels = tf.unstack(offset_image, axis=-1)
#     img_standardized = tf.stack([channels[2], channels[1], channels[0]], axis=-1)
#     return img_standardized


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
def load_initial_weights(session, weights_path):
    """
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
    as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
    dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
    need a special load function
    """

    # Load the weights into memory
    weights_dict = np.load(weights_path, encoding='bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:
        print op_name
        with tf.variable_scope(op_name, reuse=True):
            for data in weights_dict[op_name]:
                if len(data.shape) == 1:
                    var = tf.get_variable('biases', trainable=False)
                    session.run(var.assign(data))
                else:
                    var = tf.get_variable('weights', trainable=False)
                    session.run(var.assign(data))

ALEX_PATCH_DEPTH_1, ALEX_PATCH_DEPTH_2, ALEX_PATCH_DEPTH_3, ALEX_PATCH_DEPTH_4 = 96, 256, 384, 256
ALEX_PATCH_SIZE_1, ALEX_PATCH_SIZE_2, ALEX_PATCH_SIZE_3, ALEX_PATCH_SIZE_4 = 11, 5, 3, 3
ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2 = 4096, 4096
def define_variables(patch_size1 = ALEX_PATCH_SIZE_1, patch_size2 = ALEX_PATCH_SIZE_2,
                      patch_size3 = ALEX_PATCH_SIZE_3, patch_size4 = ALEX_PATCH_SIZE_4,
                      patch_depth1 = ALEX_PATCH_DEPTH_1, patch_depth2 = ALEX_PATCH_DEPTH_2,
                      patch_depth3 = ALEX_PATCH_DEPTH_3, patch_depth4 = ALEX_PATCH_DEPTH_4,
                      num_hidden1 = ALEX_NUM_HIDDEN_1, num_hidden2 = ALEX_NUM_HIDDEN_2,
                      image_width = 224, image_height = 224, image_depth = 3, num_labels = 1000):

    w1 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))

    w2 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))

    w3 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth2, patch_depth3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([patch_depth3]))

    w4 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth3], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[patch_depth3]))

    w5 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth3], stddev=0.1))
    b5 = tf.Variable(tf.zeros([patch_depth3]))

    pool_reductions = 3
    conv_reductions = 2
    no_reductions = pool_reductions + conv_reductions
    w6 = tf.Variable(tf.truncated_normal([(image_width // 2**no_reductions)*(image_height // 2**no_reductions)*patch_depth3, num_hidden1], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))

    w7 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))

    w8 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [num_labels]))

    variables = {
                 'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8,
                 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8
                }
    return variables
def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
def model_alexnet(data, variables, keep_prob):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 4, 4, 1], padding='SAME')
    layer1_relu = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer1_norm = tf.nn.local_response_normalization(layer1_pool)

    layer2_conv = tf.nn.conv2d(layer1_norm, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_relu = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer2_norm = tf.nn.local_response_normalization(layer2_pool)

    layer3_conv = tf.nn.conv2d(layer2_norm, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_relu = tf.nn.relu(layer3_conv + variables['b3'])

    layer4_conv = tf.nn.conv2d(layer3_relu, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_relu = tf.nn.relu(layer4_conv + variables['b4'])

    layer5_conv = tf.nn.conv2d(layer4_relu, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_relu = tf.nn.relu(layer5_conv + variables['b5'])
    layer5_pool = tf.nn.max_pool(layer5_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer5_norm = tf.nn.local_response_normalization(layer5_pool)

    flat_layer = flatten_tf_array(layer5_norm)
    layer6_fccd = tf.matmul(flat_layer, variables['w6']) + variables['b6']
    layer6_tanh = tf.tanh(layer6_fccd)
    layer6_drop = tf.nn.dropout(layer6_tanh, keep_prob)

    layer7_fccd = tf.matmul(layer6_drop, variables['w7']) + variables['b7']
    layer7_tanh = tf.tanh(layer7_fccd)
    layer7_drop = tf.nn.dropout(layer7_tanh, keep_prob)

    logits = tf.matmul(layer7_drop, variables['w8']) + variables['b8']
    return logits
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
        # model = pj.AlexNet(processed_image)

        # In order to get probabilities we apply softmax on the output.
        # probabilities = tf.nn.softmax(model.fc8)
        variables = define_variables()
        logits = model_alexnet(processed_image, variables, keep_prob)
        probabilities = tf.nn.softmax(logits)

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
            load_initial_weights(sess, cfg.model_weights_path)
            exit()

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

                # sorted_english = [lm.s_to_english[l] for l in pred_labels]

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
