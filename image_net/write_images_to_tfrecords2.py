# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import sys
import imp
import glob
import image_net.labels.label_maps as lm
import scipy
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def main(cfg):
    tfrecords_filename = os.path.join(cfg.output_directory, 'train-0001')

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    # Let's collect the real images to later on compare
    # to the reconstructed ones
    pattern = os.path.join(cfg.train_directory, '*', '*')

    file_names = glob.glob(pattern)
    print len(file_names)
    old_img = tf.placeholder(dtype=tf.string)
    decode = tf.image.decode_jpeg(old_img, channels=3)
    image = decode
    smallest_side = float(min(cfg.image_size))
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(
        tf.greater(height, width),
        lambda: smallest_side / width,
        lambda: smallest_side / height
    )
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    image = tf.image.resize_images(image, [new_height, new_width])
    image = tf.image.resize_image_with_crop_or_pad(image, *cfg.image_size)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    altered = image
    sess = tf.Session()
    for img_num, img_path in enumerate(file_names):
        if img_num % 1000 == 0:
            print img_num
        syn_code = img_path.split('/')[-2]
        found = False

        for key in lm.s_to_syn:
            if syn_code == lm.s_to_syn[key]:
                label = key
                found = True
                break
        if not found:
            raise ValueError, ("syn code " + syn_code + "not found")

        with tf.gfile.FastGFile(img_path, 'r') as f:
            img = f.read()

        new_img = sess.run(altered, feed_dict={old_img: img})
        scipy.misc.toimage(new_img).save('outfile.jpg')
        with tf.gfile.FastGFile('outfile.jpg', 'r') as f:
            image_data = f.read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': _bytes_feature(image_data),
            'image/class/label': _int64_feature(label),
            'image/class/syn_code': _bytes_feature(tf.compat.as_bytes(syn_code))
            }))

        writer.write(example.SerializeToString())

    writer.close()
    sess.close()
if __name__ == '__main__':
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
