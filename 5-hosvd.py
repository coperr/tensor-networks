import numpy as np

# The goal of this exercise is to first form a low-rank Tucker decomposition with exponentially decaying core (singular) values, form the full tensor to then recompute an approximate low-rank decomposition minizing the rank for a given epsilon global error threshold.

# Parameters
N = 50          # Full tensor is of size N x N x N
R_target = 10   # Target rank for the generated Tucker decomposition. Core tensor is of size R_target x R_target x R_target
epsilon = 1e-5  # Rank truncation global error threshold. Adjust this to see what ranks you obtain at different precision levels; 5e-1, 1e-1, 1e-2, 1e-3, ..., 1e-15
np.random.seed(42)

# Generate a "Decaying yet Full-Rank" Core
coords = np.ogrid[:R_target, :R_target, :R_target]
# Dominant decaying structure
G_decay = 0.25**(coords[0] + coords[1] + coords[2])
# Random "full-rank" perturbation
G_rand = np.random.randn(R_target, R_target, R_target)
# Final Core
G_true = G_decay * G_rand

# Generate orthogonal factor matrices
def random_orthonormal(N, R):
    Q, _ = np.linalg.qr(np.random.randn(N, R))
    return Q[:, :R]

A_true = random_orthonormal(N, R_target)
B_true = random_orthonormal(N, R_target)
C_true = random_orthonormal(N, R_target)

# Form the full tensor
T = np.einsum("abc,ia,jb,kc->ijk", G_true, A_true, B_true, C_true)
norm_T = np.linalg.norm(T)

# Perform ST-HOSVD with epsilon truncation
# We choose ranks r_n so that tail energy <= (epsilon^2 / 3) * ||T||_F^2 per mode.
def unfold(X, mode):
    # mode-0: (I, J*K), mode-1: (J, I*K), mode-2: (K, I*J)
    if mode == 0:
        return X.reshape(X.shape[0], -1)
    if mode == 1:
        return np.transpose(X, (1, 0, 2)).reshape(X.shape[1], -1)
    if mode == 2:
        return np.transpose(X, (2, 0, 1)).reshape(X.shape[2], -1)
    raise ValueError("mode must be 0, 1, or 2")

def mode_n_product(X, U_T, mode):
    # U_T: (r, In). Computes X ×_mode U_T and keeps mode ordering
    Y = np.tensordot(U_T, X, axes=(1, mode))   # new axis at front
    Y = np.moveaxis(Y, 0, mode)               # put new axis back at 'mode'
    return Y

def choose_rank_from_svals(s, tail_energy_target):
    # Smallest r such that sum_{i>r} s_i^2 <= target
    tail_sq = np.cumsum(s[::-1] ** 2)[::-1]   # tail_sq[r] = sum_{i>=r} s_i^2
    for r in range(1, len(s) + 1):
        if (tail_sq[r] if r < len(s) else 0.0) <= tail_energy_target:
            return r
    return len(s)

tail_target = (epsilon ** 2 / 3.0) * (norm_T ** 2)

X = T.copy()

# Compute factor A
U, s, _ = np.linalg.svd(unfold(X, 0), full_matrices=False)
r1 = choose_rank_from_svals(s, tail_target)
A = U[:, :r1]
X = mode_n_product(X, A.T, 0)

# Compute factor B
U, s, _ = np.linalg.svd(unfold(X, 1), full_matrices=False)
r2 = choose_rank_from_svals(s, tail_target)
B = U[:, :r2]
X = mode_n_product(X, B.T, 1)

# Compute factor C and G
U, s, _ = np.linalg.svd(unfold(X, 2), full_matrices=False)
r3 = choose_rank_from_svals(s, tail_target)
C = U[:, :r3]
G = mode_n_product(X, C.T, 2)

# Recompute full approximation [[G; A, B, C]] and display ranks and relative error of approximation
T_approx = np.einsum("abc,ia,jb,kc->ijk", G, A, B, C)
rel_error = np.linalg.norm(T - T_approx) / norm_T

print("Chosen Tucker ranks (r1, r2, r3) =", (r1, r2, r3))
print("Core shape:", G.shape)
print("Relative error ||T - T_approx|| / ||T|| =", rel_error)
print("Target epsilon =", epsilon)
