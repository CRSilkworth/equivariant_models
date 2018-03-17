
import glob
import pprint
import os
import sys
import imp


import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import image_net.labels.label_maps as lm

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      'weights', 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

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
            # label_num = lm.l_to_s[img_labels[line_num]]
            label_num = img_labels[line_num]
            label_english = lm.s_to_english[label_num]
        yield img_path, label_num, label_english
def main(cfg):
    create_graph()

    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

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
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)


            top_k = predictions.argsort()[-5:][::-1]
            print top_k, label_english
            if label_num in top_k:
                right += 1
            else:
                wrong += 1

            if image_num % 10 == 0:
                print right/float(right + wrong)

if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
