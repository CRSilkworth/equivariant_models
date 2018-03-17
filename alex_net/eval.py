import os
import cv2
import numpy as np
import tensorflow as tf
import image_net.image_generator as ig
import image_net.labels.label_maps as lm
import alex_net.kratzert as kr
import os
import sys
import imp
import glob
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


def main(cfg):


    image_file_name_placeholder = tf.placeholder(tf.string)

    processed_image = preprocess_image(
        image_file_name_placeholder,
        cfg.image_size[0],
        cfg.image_size[1],
        3,
        cfg.mean_rgb
    )

    keep_prob = tf.placeholder(tf.float32)

    model = kr.AlexNet(processed_image, keep_prob, 1000, [], weights_path=cfg.model_weights_path)
    probs = tf.nn.softmax(model.fc8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        right = 0
        wrong = 0

        img_label_iter = image_paths_and_labels(
            cfg.img_dir,
            cfg.img_label_file_name,
            '../image_net/labels/synsets.txt'
            # cfg.label_def_file_name
        )

        for img_num, img_label in enumerate(img_label_iter):
            image_path, label_num, label_english = img_label

            probs_e = sess.run(probs, feed_dict={image_file_name_placeholder: image_path, keep_prob: 1.})

            pred_label_num = np.argmax(probs_e)

            if pred_label_num == label_num:
                right += 1
            else:
                wrong += 1

            if img_num % 100 == 0:
                print float(right)/(wrong+right)
if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
