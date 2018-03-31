import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import image_net.image_transformations as da
import scipy

class TestImageTransformations(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.mean_rgb = np.array([121.65, 116.67, 102.82], dtype=np.float32)
        self.rgb_eigenvectors = np.array(
            [
                [4.0456429e-01, -7.1764714e-01, -5.6684244e-01],
                [-8.1423664e-01, -4.7433414e-04, -5.8053291e-01],
                [4.1634887e-01, 6.9640672e-01, -5.8452654e-01],
            ],
            dtype=np.float32
        )
        self.rgb_eigenvalues = np.array(
            [258.35184, 1167.8958, 13727.228],
            dtype=np.float32
        )

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_rgb_distortion(self):
        file_names = ['test_images/n02120505_9789.JPEG', 'test_images/n03787032_9989.JPEG', 'test_images/n04162706_9899.JPEG']
        to_run = {}
        with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                encoded = tf.placeholder(dtype=tf.string)
                image = tf.image.decode_jpeg(encoded, channels=3)
                image = tf.to_float(image)
                to_run['before'] = image

                image = da.rgb_distortion(image, self.rgb_eigenvectors, self.rgb_eigenvalues, stddev=0.3, to_run=to_run)
                to_run['image'] = image
            with tf.Session() as sess:
                for img_num, img_path in enumerate(file_names):
                    with tf.gfile.FastGFile(img_path, 'r') as f:
                        img = f.read()
                    new_path = img_path.replace('.JPEG', '_distorted.JPEG')
                    results = sess.run(to_run, feed_dict={encoded: img})

                    # print results['before'] - results['after']
                    scipy.misc.toimage(results['image']).save(new_path)

if __name__ == "__main__":
    unittest.main()
