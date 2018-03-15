import os
import cv2
import numpy as np
import tensorflow as tf
import image_net.image_generator as ig
import image_net.labels.image_net_to_kratzert as ik
import alex_net.kratzert as kr
import os
import sys
import imp

def main(cfg):

    img_gen = ig.ImageGenerator(
        img_dir=cfg.img_dir,
        img_label_file_name=cfg.img_label_file_name,
        allowed_extensions=cfg.allowed_extensions,
        size=cfg.image_size,
        label_file_name=cfg.label_file_name,
    )

    img_placeholder = tf.placeholder(tf.float32, [1, cfg.image_size[0], cfg.image_size[1], 3])
    keep_prob = tf.placeholder(tf.float32)

    model = kr.AlexNet(img_placeholder, keep_prob, 1000, [])
    probs = tf.nn.softmax(model.fc8)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        right = 0
        wrong = 0
        for img_num, (img, label_num, label_english) in enumerate(img_gen):

            probs_e = sess.run(probs, feed_dict={img_placeholder: img, keep_prob: 1.})

            pred_label_num = np.argmax(probs_e)
            mapped_label_num = ik.im_to_kr[label_num]

            if pred_label_num == mapped_label_num:
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
