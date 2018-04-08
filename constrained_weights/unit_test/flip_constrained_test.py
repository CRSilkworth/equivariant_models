import tensorflow as tf
import unittest
import numpy as np
import random
import string
# import constrained_weights
import constrained_weights.flip_constrained as fc
class TestConstrainedWeights(unittest.TestCase):
    def test_flip_equivariant_weights(self):
        with tf.variable_scope('1'):
            shape1 = [4, 4]
            weights1 = fc.flip_equivariant_weights(shape1, in_axis=1, out_axis=0)
        with tf.variable_scope('2'):
            shape2 = [5, 3]
            weights2 = fc.flip_equivariant_weights(shape2, in_axis=1, out_axis=0)
        with tf.variable_scope('3'):
            shape3 = [4, 5]
            weights3 = fc.flip_equivariant_weights(shape3, in_axis=1, out_axis=0)
        with tf.variable_scope('4'):
            shape4 = [4, 5, 2]
            weights4 = fc.flip_equivariant_weights(shape4, in_axis=1, out_axis=0)
        with tf.variable_scope('5'):
            shape5 = [4, 5, 3]
            weights5 = fc.flip_equivariant_weights(shape5, out_axis=1, in_axis=2)

        with tf.variable_scope('6'):
            shape6 = [4, 5, 3, 12]
            weights6 = fc.flip_equivariant_weights(shape6, out_axis=1, in_axis=3)


        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        weights1_e = sess.run(weights1)
        self._equivariant_satisfied(weights1_e, 0, 1)
        self._assert_equivariant(weights1_e, 0, 1)

        weights2_e = sess.run(weights2)
        self._equivariant_satisfied(weights2_e, 0, 1)
        self._assert_equivariant(weights2_e, 0, 1)

        weights3_e = sess.run(weights3)
        self._equivariant_satisfied(weights3_e, 0, 1)
        self._assert_equivariant(weights3_e, 0, 1)

        weights4_e = sess.run(weights4)
        self._equivariant_satisfied(weights4_e, 0, 1)

        weights5_e = sess.run(weights5)
        self._equivariant_satisfied(weights5_e, 2, 1)

        weights6_e = sess.run(weights6)
        self._equivariant_satisfied(weights6_e, 1, 3)
        self._assert_equivariant(weights6_e, 1, 3)

    def test_flip_equivariant_layer(self):

        sess = tf.Session()

        shape_in = [2, 3, 1]
        shape_out = [1, 2, 3]
        flip_axis = 1
        self._test_wrapper(sess, shape_in, shape_out, flip_axis)

        # shape_in = [2, 3, 2]
        # shape_out = [3, 4, 5]
        # flip_axis = 1
        # self._test_wrapper(sess, shape_in, shape_out, flip_axis)
        #
        # shape_in = [3, 4, 2]
        # shape_out = [3, 4, 5]
        # flip_axis = 0
        # self._test_wrapper(sess, shape_in, shape_out, flip_axis)

    def test_flip_invariant_weights(self):
        with tf.variable_scope('1i'):
            shape1 = [4, 4]
            weights1 = fc.flip_invariant_weights(shape1, in_axis=1, out_axis=0)
        with tf.variable_scope('2i'):
            shape2 = [5, 3]
            weights2 = fc.flip_invariant_weights(shape2, in_axis=1, out_axis=0)
        with tf.variable_scope('3i'):
            shape3 = [4, 5]
            weights3 = fc.flip_invariant_weights(shape3, in_axis=1, out_axis=0)
        with tf.variable_scope('4i'):
            shape4 = [4, 5, 2]
            weights4 = fc.flip_invariant_weights(shape4, in_axis=1, out_axis=0)
        with tf.variable_scope('5i'):
            shape5 = [4, 5, 3]
            weights5 = fc.flip_invariant_weights(shape5, out_axis=1, in_axis=2)

        with tf.variable_scope('6i'):
            shape6 = [4, 5, 3, 12]
            weights6 = fc.flip_invariant_weights(shape6, out_axis=1, in_axis=3)


        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        weights1_e = sess.run(weights1)
        self._invariant_satisfied(weights1_e, 0, 1)
        self._assert_invariant(weights1_e, 0, 1)
        #
        weights2_e = sess.run(weights2)
        self._invariant_satisfied(weights2_e, 0, 1)
        self._assert_invariant(weights2_e, 0, 1)
        #
        weights3_e = sess.run(weights3)
        self._invariant_satisfied(weights3_e, 0, 1)
        self._assert_invariant(weights3_e, 0, 1)
        #
        weights4_e = sess.run(weights4)
        self._invariant_satisfied(weights4_e, 0, 1)
        #
        weights5_e = sess.run(weights5)
        self._invariant_satisfied(weights5_e, 1, 2)
        #
        weights6_e = sess.run(weights6)
        self._invariant_satisfied(weights6_e, 1, 3)
        self._assert_invariant(weights6_e, 1, 3)
    def _test_wrapper(self, sess, shape_in, shape_out, flip_axis):
        r_string = ''.join([random.choice(string.ascii_lowercase) for i in range(5)])
        with tf.variable_scope(r_string):
            input = self._create_input(shape_in, flip_axis)
            output = fc.flip_equivariant_layer(input, shape_out, flip_axis)

        sess.run(tf.global_variables_initializer())
        input_e, output_e = sess.run([input, output])
        # output_e = sess.run(input)
        # print input_e[0].flatten(), input_e[1].flatten()
        # print output_e
        # self.assertTrue(
        #     self._is_close(output_e[0], np.flip(output_e[1], axis=flip_axis))
        # )

    def _create_input(self, shape, flip_axis):

        random = np.random.rand(np.prod(shape)).astype(np.float32)
        input = tf.reshape(
            np.array(range(np.prod(shape))).astype(np.float32),
            # random,
            [1] + shape
        )

        input_flipped = tf.reverse(input, axis=[flip_axis + 1])
        input = tf.concat([input, input_flipped], axis=0)
        return input

    def _assert_equivariant(self, weights, out_axis, in_axis):
        shape = weights.shape
        out_dim = shape[out_axis]
        in_dim = shape[in_axis]

        if len(shape) % 2 != 0:
            return
        input_shape = shape[len(shape)/2:]
        input = np.random.rand(*input_shape)
        input_flipped = np.flip(input, axis=in_axis - len(shape)/2)

        axes = [range(len(shape)/2, len(shape)), range(len(shape)/2)]
        mul = np.tensordot(weights, input, axes=axes)
        mul_flipped = np.tensordot(weights, input_flipped, axes=axes)

        self.assertTrue(
            self._is_close(mul, np.flip(mul_flipped, axis=out_axis))
        )
    def _assert_invariant(self, weights, out_axis, in_axis):
        shape = weights.shape
        out_dim = shape[out_axis]
        in_dim = shape[in_axis]

        if len(shape) % 2 != 0:
            return
        input_shape = shape[len(shape)/2:]
        input = np.random.rand(*input_shape)
        input_flipped = np.flip(input, axis=in_axis - len(shape)/2)

        axes = [range(len(shape)/2, len(shape)), range(len(shape)/2)]
        mul = np.tensordot(weights, input, axes=axes)
        mul_flipped = np.tensordot(weights, input_flipped, axes=axes)

        self.assertTrue(
            self._is_close(mul, mul_flipped)
        )
    def _is_close(self, array1, array2):
        diff = np.abs(array1 - array2)
        return np.all(diff < 0.0001)
    def _equivariant_satisfied(self, weights, out_axis, in_axis):
        shape = weights.shape
        out_dim = shape[out_axis]
        in_dim = shape[in_axis]
        half_rows = int(shape[0]/2. + 0.5)

        for row in range(half_rows):
            for col in range(shape[1]):
                if row >= out_dim or col >= in_dim:
                    continue
                if out_dim - row - 1 >= shape[out_axis] or in_dim - col - 1 >= shape[in_axis]:
                    continue

                weights_1 = self._get_slices(weights, row, col, shape, out_axis, in_axis)

                weights_2 = self._get_slices(weights, out_dim - row - 1, in_dim - col - 1, shape, out_axis, in_axis)

                # if not weights_1.shape:
                    # self.assertEquals(weights_1, weights_2)
                # else:
                self.assertTrue(np.all(weights_1 == weights_2))

    def _invariant_satisfied(self, weights, out_axis, in_axis):
        shape = weights.shape
        out_dim = shape[out_axis]
        in_dim = shape[in_axis]
        half_rows = int(shape[0]/2. + 0.5)

        for row in range(half_rows):
            for col in range(shape[1]):
                if row >= out_dim or col >= in_dim:
                    continue
                if out_dim - row - 1 >= shape[out_axis] or in_dim - col - 1 >= shape[in_axis]:
                    continue

                weights_1 = self._get_slices(weights, row, col, shape, out_axis, in_axis)

                weights_2 = self._get_slices(weights, row, in_dim - col - 1, shape, out_axis, in_axis)

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
