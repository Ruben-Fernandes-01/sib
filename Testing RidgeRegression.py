from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.linear_regression import RidgeRegression

CPU_PATH = "/Users/rubenfernandes/sib/datasets/cpu/cpu.csv"

# 1. carregar o dataset (com nomes das features; a última coluna é o y)
dataset = read_csv(CPU_PATH, sep=",", features=True, label=True)
print("Dataset:", dataset.shape())

# 2. dividir em treino e teste
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
print("Treino:", train.shape(), "| Teste:", test.shape())

# 3. treinar o modelo
model = RidgeRegression(l2_penalty=1.0, alpha=0.1, max_iter=5000,
                        patience=100, scale=True)
model.fit(train)

# score (MSE) e custo
print("\nIterações:", len(model.cost_history))
print("Custo inicial :", model.cost_history[0])
print("Custo final   :", model.cost_history[len(model.cost_history) - 1])
print("\nScore (MSE) no treino:", model.score(train))
print("Score (MSE) no teste :", model.score(test))
print("Custo no treino      :", model.cost(train))
print("Custo no teste       :", model.cost(test))