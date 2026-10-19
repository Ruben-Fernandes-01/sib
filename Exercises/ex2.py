import numpy as np
from si.data.dataset import Dataset

X = np.array([[1.0, 2.0, 3.0],
              [4.0, np.nan, 6.0],
              [7.0, 8.0, 9.0]])
y = np.array([0, 1, 0])
ds = Dataset(X, y, features=['a', 'b', 'c'], label='y')

# dropna
ds1 = Dataset(X.copy(), y.copy(), features=['a', 'b', 'c'], label='y')
ds1.dropna()
print("dropna:", ds1.X, ds1.y)

# fillna com valor fixo
ds2 = Dataset(X.copy(), y.copy(), features=['a', 'b', 'c'], label='y')
ds2.fillna(0)
print("fillna(0):", ds2.X)

# fillna com média
ds3 = Dataset(X.copy(), y.copy(), features=['a', 'b', 'c'], label='y')
ds3.fillna("mean")
print("fillna(mean):", ds3.X)

# remove_by_index
ds4 = Dataset(X.copy(), y.copy(), features=['a', 'b', 'c'], label='y')
ds4.remove_by_index(1)
print("remove_by_index(1):", ds4.X, ds4.y)