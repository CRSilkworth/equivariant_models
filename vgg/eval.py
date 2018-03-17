import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim.nets
import image_net.labels.label_to_synset as ls
import image_net.image_generator as ig
import alex_net.kratzert as kr
import os
import sys
import imp
import matplotlib.pyplot as plt
from PIL import Image
def main(cfg):

    img_gen = ig.ImageGenerator(
        img_dir=cfg.img_dir,
        img_label_file_name=cfg.img_label_file_name,
        allowed_extensions=cfg.allowed_extensions,
        mean_rgb=cfg.mean_rgb,
        size=cfg.image_size,
        label_file_name=cfg.label_file_name,
        resize_method='crop'
    )

    img_placeholder = tf.placeholder(tf.float32, [1, cfg.image_size[0], cfg.image_size[1], 3])
    keep_prob = tf.placeholder(tf.float32)

    model = tf.contrib.slim.nets.vgg
    logits, _ = model.vgg_16(img_placeholder, num_classes=1000, is_training=False)
    probs = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, cfg.model_weights_path)

        right = 0
        wrong = 0
        fig = plt.figure(figsize=(15, 6))
        for img_num, (img_path, img, label_num, label_english) in enumerate(img_gen):

            # if img_num < 5:
                # image = Image.fromarray(img[0], 'RGB')
                # image.save(img_path.split('/')[-1] + '.png')
                # image.show()
            # else:
                # break
            probs_e = sess.run(probs, feed_dict={img_placeholder: img, keep_prob: 1.})

            # pred_label_num = np.argmax(probs_e)
            pred_label_nums = np.flip(np.argsort(probs_e)[0][-5:], axis=0)

            mapped_label_nums = [ls.s_to_l[pred_label_num] for pred_label_num in pred_label_nums]

            if label_num in mapped_label_nums:
                right += 1
            else:
                wrong += 1

            print img_path, img_gen.label_map[label_num], [img_gen.label_map[mapped_label_num] for mapped_label_num in mapped_label_nums]
            # print img_path, img_gen.label_map[mapped_label_num]

            if img_num % 10 == 0:
                print float(right)/(wrong+right)
if __name__ == "__main__":
    # Read in config file
    assert len(sys.argv) == 2, "Must pass exactly one argument to the script, namely the cfg file, got " + str(len(sys.argv))
    abs_path = os.path.abspath(sys.argv[1])
    cfg = imp.load_source('*', abs_path)
    main(cfg)
