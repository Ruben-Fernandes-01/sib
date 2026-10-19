import unittest

import numpy as np

from si.neural_networks.activation import (
    ActivationLayer,
    ReLUActivation,
    SigmoidActivation,
    TanhActivation,
    SoftmaxActivation,
)


class TestActivations(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-2.5, -1.2, -0.3, 0.4, 1.1, 2.7])
        self.eps = 1e-6

    def _gradient_check(self, activation):
        numeric = (activation.activation_function(self.x + self.eps)
                   - activation.activation_function(self.x - self.eps)) / (2 * self.eps)
        analytic = activation.derivative(self.x)
        self.assertLess(np.abs(numeric - analytic).max(), 1e-8)

    def test_relu_gradient_check(self):
        self._gradient_check(ReLUActivation())

    def test_sigmoid_gradient_check(self):
        self._gradient_check(SigmoidActivation())

    def test_tanh_gradient_check(self):
        self._gradient_check(TanhActivation())

    def test_relu_values(self):
        out = ReLUActivation().activation_function(self.x)
        expected = np.array([0, 0, 0, 0.4, 1.1, 2.7])
        np.testing.assert_allclose(out, expected)

    def test_sigmoid_range(self):
        out = SigmoidActivation().activation_function(self.x)
        self.assertTrue(np.all((out > 0) & (out < 1)))

    def test_tanh_range(self):
        out = TanhActivation().activation_function(self.x)
        self.assertTrue(np.all((out > -1) & (out < 1)))

    def test_softmax_sums_to_one(self):
        out = SoftmaxActivation().activation_function(self.x)
        self.assertAlmostEqual(out.sum(), 1.0, places=10)

    def test_softmax_numerically_stable_with_large_values(self):
        big = np.array([1000.0, 1001.0, 1002.0])
        out = SoftmaxActivation().activation_function(big)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertAlmostEqual(out.sum(), 1.0, places=10)

    def test_softmax_2d_batch(self):
        X = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        out = SoftmaxActivation().activation_function(X)
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0])

    def test_forward_propagation_stores_input_and_output(self):
        act = ReLUActivation()
        out = act.forward_propagation(self.x, training=False)
        np.testing.assert_array_equal(act.input, self.x)
        np.testing.assert_array_equal(act.output, out)

    def test_backward_propagation_elementwise(self):
        act = SigmoidActivation()
        act.forward_propagation(self.x, training=True)
        output_error = np.ones_like(self.x)
        input_error = act.backward_propagation(output_error)
        expected = act.derivative(self.x) * output_error
        np.testing.assert_allclose(input_error, expected)

    def test_output_shape_matches_input_shape(self):
        act = ReLUActivation()
        act.set_input_shape((6,))
        self.assertEqual(act.output_shape(), (6,))

    def test_parameters_is_zero(self):
        self.assertEqual(ReLUActivation().parameters(), 0)


if __name__ == "__main__":
    unittest.main()