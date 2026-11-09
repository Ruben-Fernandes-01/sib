import unittest

import numpy as np

from si.data.dataset import Dataset
from si.model_selection.randomized_search import randomized_search_cv, log_uniform


class DummyModel:
    def __init__(self, c=0.0, lr=0.0):
        self.c = c
        self.lr = lr

    def fit(self, dataset):
        return self

    def predict(self, dataset):
        return np.full(dataset.X.shape[0], self.c)

    def score(self, dataset):
        return float(np.mean(self.predict(dataset) == dataset.y))


class TestLogUniform(unittest.TestCase):

    def test_samples_within_range(self):
        sampler = log_uniform(1e-4, 1e-1)
        rng = np.random.default_rng(0)
        samples = np.array([sampler(rng) for _ in range(1000)])
        self.assertTrue(np.all(samples >= 1e-4) and np.all(samples <= 1e-1))

    def test_is_uniform_in_log_scale(self):
        sampler = log_uniform(1e-4, 1e-1)
        rng = np.random.default_rng(0)
        samples = np.array([sampler(rng) for _ in range(6000)])
        # 3 décadas -> cerca de 1/3 das amostras em cada uma
        for lo, hi in [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1)]:
            frac = np.mean((samples >= lo) & (samples < hi))
            self.assertAlmostEqual(frac, 1 / 3, delta=0.04)


class TestRandomizedSearchCV(unittest.TestCase):

    def setUp(self):
        X = np.arange(40, dtype=float).reshape(20, 2)
        y = np.array([1.0] * 15 + [0.0] * 5)
        self.ds = Dataset(X, y)
        self.dist = {'c': [0.0, 1.0], 'lr': log_uniform(1e-4, 1e-1)}

    def test_returns_same_structure_as_grid_search(self):
        r = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=7, seed=0)
        self.assertEqual(set(r), {'hyperparameters', 'scores', 'best_hyperparameters', 'best_score'})
        self.assertEqual(len(r['hyperparameters']), 7)
        self.assertEqual(len(r['scores']), 7)

    def test_samples_from_list_and_function(self):
        r = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=30, seed=0)
        for h in r['hyperparameters']:
            self.assertIn(h['c'], [0.0, 1.0])
            self.assertTrue(1e-4 <= h['lr'] <= 1e-1)

    def test_best_is_highest_score(self):
        r = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=20, seed=0)
        self.assertEqual(r['best_score'], max(r['scores']))
        self.assertEqual(r['best_hyperparameters']['c'], 1.0)

    def test_reproducible_with_seed(self):
        a = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=5, seed=11)
        b = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=5, seed=11)
        self.assertEqual(a['hyperparameters'], b['hyperparameters'])
        self.assertEqual(a['scores'], b['scores'])

    def test_different_seeds_give_different_configurations(self):
        a = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=5, seed=1)
        b = randomized_search_cv(DummyModel(), self.ds, self.dist, cv=4, n_iter=5, seed=2)
        self.assertNotEqual(a['hyperparameters'], b['hyperparameters'])

    def test_works_with_builder_function(self):
        r = randomized_search_cv(lambda c, lr: DummyModel(c, lr), self.ds, self.dist,
                                 cv=4, n_iter=3, seed=0)
        self.assertEqual(len(r['scores']), 3)


if __name__ == "__main__":
    unittest.main()
