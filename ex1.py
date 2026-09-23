from si.io.csv_file import read_csv

ds = read_csv("/Users/rubenfernandes/sib/datasets/iris/iris.csv",
              sep=",", features=True, label=True)

# 1.1) já carregado acima

# 1.2) penúltima variável independente
penultimate = ds.X[:, -2]
print("1.2:", penultimate.shape)

# 1.3) últimas 10 amostras — média por feature
last_10 = ds.X[-10:]
print("1.3:", last_10.mean(axis=0))

# 1.4) amostras com todas as features <= 6
mask_le6 = (ds.X <= 6).all(axis=1)
print("1.4:", ds.X[mask_le6].shape[0])

# 1.5) amostras com classe diferente de 'Iris-setosa'
mask_not_setosa = ds.y != 'Iris-setosa'
print("1.5:", ds.X[mask_not_setosa].shape[0])