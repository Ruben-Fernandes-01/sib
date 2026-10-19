"""
Exercícios da Class4_P.pdf (antes da sub-package "neural_networks").

Colocar em: scripts/exercicios_nn.py
"""
import numpy as np


# ---------------------------------------------------------------------------
# Exercício 1
# Rede com 2 entradas, 3 neurónios e 1 saída.
# Calcular com eles só a primeira camada.
# ---------------------------------------------------------------------------
print("Exercício 1")

x = np.array([1, 2])
W = np.array([[0.1, 0.2, -0.1],
              [0.3, -0.2, 0.4]])
b = np.array([0.1, 0, -0.1])

H1 = x.dot(W) + b
print("x:", x.shape, "| W:", W.shape, "| b:", b.shape)
print("H1 = x.W + b =", H1, "| shape:", H1.shape)


# ---------------------------------------------------------------------------
# Exercício 2
# X aleatório com 5 amostras e 2 features.
# Duas camadas: 2 -> 3 -> 1, pesos aleatórios em [-0.5, 0.5), bias a zero.
# Calcular a saída e imprimir a shape de cada resultado intermédio.
# ---------------------------------------------------------------------------
print("\nExercício 2")

X = np.random.rand(5, 2)
print("X:", X.shape)

# camada 1: 2 -> 3
W1 = np.random.rand(2, 3) - 0.5     # uniforme em [-0.5, 0.5)
b1 = np.zeros((1, 3))

# camada 2: 3 -> 1
W2 = np.random.rand(3, 1) - 0.5
b2 = np.zeros((1, 1))

H1 = X.dot(W1) + b1
print("H1 (saída da camada 1):", H1.shape)

Y = H1.dot(W2) + b2
print("Y (saída da camada 2):", Y.shape)