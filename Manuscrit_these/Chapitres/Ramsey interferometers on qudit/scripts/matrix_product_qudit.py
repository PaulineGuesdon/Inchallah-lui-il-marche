import numpy as np

s = 1/np.sqrt(2)

X_pi2 = s * np.array([
    [1, -1j, 0, 0],
    [-1j, 1, 0, 0],
    [0, 0, 1, -1j],
    [0, 0, -1j, 1]
], dtype=complex)          # attention au facteur 1/√2, il manquait dans ton code

X_qubit = np.array([
    [1, 0, 0, 0],
    [0, s, -1j*s, 0],
    [0, -1j*s, s, 0],
    [0, 0, 0, 1]
], dtype=complex)

def Phi(phi):
    return np.diag([1, 1, np.exp(-1j*phi), 1])

def E(k):                  # projecteur E_k, k = 1..4
    M = np.zeros((4, 4), dtype=complex)
    M[k-1, k-1] = 1
    return M

phi = 0.7                  # n'importe quelle valeur de test
U = X_qubit @ Phi(phi) @ X_pi2          # X_pi2 agit en premier (à droite)

E2p = U.conj().T @ E(2) @ U
E4p = U.conj().T @ E(4) @ U

# Tes formules de la thèse
e, em = np.exp(1j*phi), np.exp(-1j*phi)
E2_these = 0.25 * np.array([
    [1,     1j,    em,     -1j*em],
    [-1j,   1,     -1j*em, -em],
    [e,     1j*e,  1,      -1j],
    [1j*e,  -e,    1j,     1]
])
E4_these = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0.5, 0.5j],
    [0, 0, -0.5j, 0.5]
])

np.set_printoptions(precision=3, suppress=True)
print("E2' =\n", E2p)
print("E2' correct :", np.allclose(E2p, E2_these))
print("E4' correct :", np.allclose(E4p, E4_these))