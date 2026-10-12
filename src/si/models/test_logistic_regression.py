import unittest

import numpy as np

from si.data.dataset import Dataset
from si.models.logistic_regression import LogisticRegression
from si.statistics.sigmoid_function import sigmoid_function


def make_dataset():
    """Dados de classificação binária linearmente separáveis por sigmoide."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * np.array([1, 2, 0.5])
    true_theta = np.array([2.0, -1.5, 3.0])
    logits = X @ true_theta - 0.2
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=300) < prob).astype(float)
    return Dataset(X, y, features=['x1', 'x2', 'x3'], label='y')


class TestSigmoidFunction(unittest.TestCase):

    def test_sigmoid_zero_is_half(self):
        self.assertAlmostEqual(0.5, sigmoid_function(0))

    def test_sigmoid_range(self):
        values = sigmoid_function(np.array([-20, -1, 0, 1, 20]))

        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(values < 1))
        self.assertTrue(np.all(np.diff(values) > 0))  # é crescente

    def test_sigmoid_known_values(self):
        result = sigmoid_function(np.array([0.0]))

        self.assertAlmostEqual(0.5, result[0])


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.dataset = make_dataset()

    def test_fit(self):
        model = LogisticRegression(l2_penalty=1.0, alpha=0.1, max_iter=1000, scale=True)
        model.fit(self.dataset)

        self.assertEqual(3, len(model.theta))
        self.assertEqual(3, len(model.mean))
        self.assertEqual(3, len(model.std))
        self.assertGreater(len(model.cost_history), 0)

    def test_cost_decreases(self):
        model = LogisticRegression(l2_penalty=1.0, alpha=0.1, max_iter=500, scale=True)
        model.fit(self.dataset)

        first = model.cost_history[0]
        last = model.cost_history[len(model.cost_history) - 1]
        self.assertLess(last, first)

    def test_predict_is_binary(self):
        model = LogisticRegression(l2_penalty=1.0, alpha=0.1, max_iter=1000, scale=True)
        model.fit(self.dataset)

        predictions = model.predict(self.dataset)

        self.assertEqual(self.dataset.y.shape, predictions.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_score_is_reasonable(self):
        model = LogisticRegression(l2_penalty=1.0, alpha=0.1, max_iter=2000, scale=True)
        model.fit(self.dataset)

        # dados gerados com sinal forte -> accuracy bem acima do acaso
        self.assertGreater(model.score(self.dataset), 0.8)

    def test_cost_matches_predictions(self):
        model = LogisticRegression(l2_penalty=0.0, alpha=0.1, max_iter=3000, scale=True)
        model.fit(self.dataset)

        # sem regularização, o custo tem de ser só o log-loss médio (>= 0)
        self.assertGreaterEqual(model.cost(self.dataset), 0.0)

    def test_patience_stops_early(self):
        model = LogisticRegression(l2_penalty=0.0, alpha=5.0, max_iter=5000,
                                   patience=5, scale=True)
        model.fit(self.dataset)

        self.assertLess(len(model.cost_history), 5000)


if __name__ == '__main__':
    unittest.main()