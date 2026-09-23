import unittest

import numpy as np

from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile


class TestSelectPercentile(unittest.TestCase):

    def test_fit(self):
        dataset = Dataset(
            X=np.array([[1, 2, 0, 3],
                        [2, 1, 4, 3],
                        [1, 1, 1, 3],
                        [3, 3, 5, 2]]),
            y=np.array([0, 1, 0, 1]),
            features=['f1', 'f2', 'f3', 'f4'],
            label='y'
        )

        selector = SelectPercentile(percentile=50)
        selector.fit(dataset)

        self.assertEqual(4, len(selector.F))
        self.assertEqual(4, len(selector.p))

    def test_transform(self):
        dataset = Dataset(
            X=np.array([[1, 2, 0, 3],
                        [2, 1, 4, 3],
                        [1, 1, 1, 3],
                        [3, 3, 5, 2]]),
            y=np.array([0, 1, 0, 1]),
            features=['f1', 'f2', 'f3', 'f4'],
            label='y'
        )

        selector = SelectPercentile(percentile=50)
        new_dataset = selector.fit_transform(dataset)

        # 50% de 4 features -> 2 features
        self.assertEqual(2, new_dataset.X.shape[1])
        self.assertEqual(2, len(new_dataset.features))
    def test_transform_iris_like(self):
        # exemplo do enunciado: F-values conhecidos, percentile=40 -> 4 de 10 features
        dataset = Dataset(
            X=np.arange(10).reshape(1, 10).astype(float),
            features=[f"f{i}" for i in range(10)]
        )

        selector = SelectPercentile(percentile=40)
        selector.F = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
        selector.p = np.zeros_like(selector.F)
        selector.is_fitted = True

        new_dataset = selector.transform(dataset)

        self.assertEqual(4, new_dataset.X.shape[1])
        self.assertEqual(['f3', 'f5', 'f6', 'f7'], new_dataset.features)


if __name__ == '__main__':
    unittest.main()