import unittest

import numpy as np

from si.neural_networks.losses import (
    LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
)


def numeric_derivative(loss, y_true, y_pred, eps=1e-6):
    num = np.zeros_like(y_pred)
    for idx in np.ndindex(y_pred.shape):
        p, m = y_pred.copy(), y_pred.copy()
        p[idx] += eps
        m[idx] -= eps
        num[idx] = (loss.loss(y_true, p) - loss.loss(y_true, m)) / (2 * eps)
    return num


class TestLosses(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([[1.0], [0.0], [1.0], [0.0]])
        self.y_pred = np.array([[0.9], [0.2], [0.6], [0.3]])
        self.one_hot = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        self.probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

    def test_base_class_is_abstract(self):
        with self.assertRaises(TypeError):
            LossFunction()

    def test_mse_value(self):
        # (0.01 + 0.04 + 0.16 + 0.09) / 4 = 0.075
        self.assertAlmostEqual(MeanSquaredError().loss(self.y_true, self.y_pred), 0.075)

    def test_mse_derivative(self):
        d = MeanSquaredError().derivative(self.y_true, self.y_pred)
        np.testing.assert_allclose(d, 2 * (self.y_pred - self.y_true) / 4)

    def test_mse_gradient_check(self):
        L = MeanSquaredError()
        num = numeric_derivative(L, self.y_true, self.y_pred)
        np.testing.assert_allclose(num, L.derivative(self.y_true, self.y_pred), atol=1e-8)

    def test_bce_value(self):
        expected = -np.sum(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        self.assertAlmostEqual(BinaryCrossEntropy().loss(self.y_true, self.y_pred), expected)

    def test_bce_gradient_check(self):
        L = BinaryCrossEntropy()
        num = numeric_derivative(L, self.y_true, self.y_pred)
        np.testing.assert_allclose(num, L.derivative(self.y_true, self.y_pred), atol=1e-6)

    def test_bce_no_infinite_with_extreme_predictions(self):
        L = BinaryCrossEntropy()
        extreme = np.array([[1.0], [0.0], [1.0], [0.0]])
        self.assertTrue(np.isfinite(L.loss(self.y_true, extreme)))
        self.assertTrue(np.all(np.isfinite(L.derivative(self.y_true, extreme))))

    def test_cce_value(self):
        expected = -(np.log(0.7) + np.log(0.8) + np.log(0.6))
        self.assertAlmostEqual(CategoricalCrossEntropy().loss(self.one_hot, self.probs), expected)

    def test_cce_gradient_check(self):
        L = CategoricalCrossEntropy()
        num = numeric_derivative(L, self.one_hot, self.probs)
        np.testing.assert_allclose(num, L.derivative(self.one_hot, self.probs), atol=1e-6)

    def test_cce_no_infinite_with_zero_probability(self):
        L = CategoricalCrossEntropy()
        probs = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertTrue(np.isfinite(L.loss(self.one_hot, probs)))
        self.assertTrue(np.all(np.isfinite(L.derivative(self.one_hot, probs))))


if __name__ == "__main__":
    unittest.main()
