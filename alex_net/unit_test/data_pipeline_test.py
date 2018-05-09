import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import alex_net.data_pipeline as dp
import scipy
from PIL import Image
import numpy as np

class TestDataPipeline(unittest.TestCase):
    def test_original(self):
        with tf.Graph().as_default() as graph:
            original_dataset = dp.AlexNetDataset(
                file_names=['test_images/tf_records_images/validation-00000-of-00001'],
                batch_size=1,
                num_epochs=1,
                shuffle_buffer_size=1,
                image_size=[256, 256],
                rgb_mean=[0., 0., 0.],
                keep_prob=1.,
                num_threads=1
            )

            # Create the iterator shared by all three dataset
            data_iter = original_dataset.reinitializable_iterator()

            # Create the initializers of the datasets
            original_initializer = original_dataset.iterator_initializer(data_iter)

            images, labels, keep_prob = data_iter.get_next()
            images = tf.cast(images, tf.uint8)

            encoded = tf.image.encode_jpeg(images[0], format='rgb')
            # Define the variable initializer operation
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            with tf.Session() as sess:
                sess.run(init_op)

                sess.run(original_initializer)
                for i in xrange(3):
                    images_eval = sess.run(encoded)
                    self._write_image(images_eval, 'test_images/' + str(i) + '_original.jpg')

    def test_flipped(self):
        with tf.Graph().as_default() as graph:
            flipped_dataset = dp.AlexNetDataset(
                file_names=['test_images/tf_records_images/validation-00000-of-00001'],
                batch_size=1,
                num_epochs=1,
                shuffle_buffer_size=1,
                image_size=[256, 256],
                rgb_mean=[0., 0., 0.],
                keep_prob=1.,
                num_threads=1
            )
            flipped_dataset.horizontal_flip()

            data_iter = flipped_dataset.reinitializable_iterator()

            # Create the initializers of the datasets
            flipped_initializer = flipped_dataset.iterator_initializer(data_iter)

            images, labels, keep_prob = data_iter.get_next()
            images = tf.cast(images, tf.uint8)

            images = tf.unstack(images[0])
            encoded = []
            for image in images:
                encoded.append(tf.image.encode_jpeg(image, format='rgb'))
            # Define the variable initializer operation
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            with tf.Session() as sess:
                sess.run(init_op)

                sess.run(flipped_initializer)
                for i in xrange(3):
                    images_eval = sess.run(encoded)
                    for num, image_eval in enumerate(images_eval):
                        self._write_image(image_eval, 'test_images/' + str(i) + '_' + str(num) + '_flipped.jpg')

    def test_cropped(self):
        with tf.Graph().as_default() as graph:
            cropped_dataset = dp.AlexNetDataset(
                file_names=['test_images/tf_records_images/validation-00000-of-00001'],
                batch_size=1,
                num_epochs=1,
                shuffle_buffer_size=1,
                image_size=[256, 256],
                rgb_mean=[0., 0., 0.],
                keep_prob=1.,
                num_threads=1
            )
            cropped_dataset.five_crops([224, 224])

            # Create the iterator shared by all three dataset
            data_iter = cropped_dataset.reinitializable_iterator()

            # Create the initializers of the datasets
            cropped_initializer = cropped_dataset.iterator_initializer(data_iter)

            images, labels, keep_prob = data_iter.get_next()
            images = tf.cast(images, tf.uint8)

            images = tf.unstack(images[0])
            encoded = []
            for image in images:
                encoded.append(tf.image.encode_jpeg(image, format='rgb'))
            # Define the variable initializer operation
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            with tf.Session() as sess:
                sess.run(init_op)

                sess.run(cropped_initializer)
                for i in xrange(3):
                    images_eval = sess.run(encoded)
                    for num, image_eval in enumerate(images_eval):
                        self._write_image(image_eval, 'test_images/' + str(i) + '_' + str(num) + '_cropped.jpg')
    def _write_image(self, array, file_name):
        with open(file_name, 'w') as f:
            f.write(array)
        # img = Image.fromarray(array, 'RGB')
        # img.save(file_name)



if __name__ == "__main__":
    unittest.main()
