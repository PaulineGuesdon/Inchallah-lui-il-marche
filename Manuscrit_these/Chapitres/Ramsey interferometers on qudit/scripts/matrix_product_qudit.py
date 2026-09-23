import numpy as np

# Création de deux matrices 4x4
pi_sur_2 = np.array([
    [1, -1j, 0, 0],
    [-1j, 1, 0, 0],
    [0, 0, 1, -1j],
    [0, 0, -1j, 1]
], dtype=complex)

qubit = np.array([
    [1, 0, 0, 0],
    [0, (1/np.sqrt(2)), -1j*(1/np.sqrt(2)), 0],
    [0, -1j*(1/np.sqrt(2)), (1/np.sqrt(2)), 0],
    [0, 0, 0, 1]
], dtype=complex)

phi = np.diag(1, 1, np.exp(-1j*p), 1)
sept = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=complex)


# Opérations de base
print("A + B =\n", A + B)
print("A * 2 =\n", A * 2)
print("A x R (produit matriciel) =\n", A @ R)
print("Transposée de A =\n", A.T)

# Déterminant et inverse
print("det(A) =", np.linalg.det(A))
print("Inverse de A =\n", np.linalg.inv(A))

# Valeurs propres et vecteurs propres
valeurs, vecteurs = np.linalg.eig(A)
print("Valeurs propres :", valeurs)
print("Vecteurs propres (en colonnes) :\n", vecteurs)

# Résoudre A x = b
b = np.array([1, 2, 3, 4], dtype=float)
x = np.linalg.solve(A, b)
print("Solution de Ax = b :", x)