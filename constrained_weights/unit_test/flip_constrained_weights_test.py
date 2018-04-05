import tensorflow as tf
import unittest
import numpy as np
# import constrained_weights
import constrained_weights.flip_constrained_weights as fc
class TestConstrainedWeights(unittest.TestCase):
    def test_flip_equivariant(self):
        with tf.variable_scope('1'):
            shape1 = [4, 4]
            weights1 = fc.flip_equivariant(shape1, in_axis=1, out_axis=0)
        with tf.variable_scope('2'):
            shape2 = [5, 3]
            weights2 = fc.flip_equivariant(shape2, in_axis=1, out_axis=0)
        with tf.variable_scope('3'):
            shape3 = [4, 5]
            weights3 = fc.flip_equivariant(shape3, in_axis=1, out_axis=0)
        with tf.variable_scope('4'):
            shape4 = [4, 5, 2]
            weights4 = fc.flip_equivariant(shape4, in_axis=1, out_axis=0)
        with tf.variable_scope('5'):
            shape5 = [4, 5, 3]
            weights5 = fc.flip_equivariant(shape5, out_axis=2, in_axis=1)

        with tf.variable_scope('6'):
            shape6 = [4, 5, 3, 12]
            weights6 = fc.flip_equivariant(shape6, out_axis=1, in_axis=3)


        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        weights1_e = sess.run(weights1)
        self._satisfied(weights1_e, shape1, 0, 1)

        weights2_e = sess.run(weights2)
        self._satisfied(weights2_e, shape2, 0, 1)

        weights3_e = sess.run(weights3)
        self._satisfied(weights3_e, shape3, 0, 1)

        weights4_e = sess.run(weights4)
        self._satisfied(weights4_e, shape4, 0, 1)

        weights5_e = sess.run(weights5)
        self._satisfied(weights5_e, shape5, 2, 1)

        weights6_e = sess.run(weights6)
        self._satisfied(weights6_e, shape6, 1, 3)

    def _satisfied(self, weights, shape, out_axis, in_axis):
        max_dim = max(shape)
        half_rows = int(shape[0]/2. + 0.5)

        for row in range(half_rows):
            for col in range(shape[1]):
                if row >= shape[out_axis] or col >= shape[in_axis]:
                    continue
                if max_dim - row - 1 >= shape[out_axis] or max_dim - col - 1 >= shape[in_axis]:
                    continue

                weights_1 = self._get_slices(weights, row, col, shape, out_axis, in_axis)

                weights_2 = self._get_slices(weights, max_dim - row - 1, max_dim - col - 1, shape, out_axis, in_axis)

                # if not weights_1.shape:
                    # self.assertEquals(weights_1, weights_2)
                # else:
                self.assertTrue(np.all(weights_1 == weights_2))

    def _get_slices(self, weights, row, col, shape, out_axis, in_axis):
        irrelevant_axes = [axis for axis, dim in enumerate(shape) if axis not in (in_axis, out_axis)]
        new_weights = np.transpose(weights, [out_axis, in_axis] + irrelevant_axes)

        return new_weights[row, col]


if __name__ == "__main__":
    unittest.main()
