import numpy as np

# The goal of this exercise is to first form a low-rank Tucker decomposition with exponentially decaying core (singular) values, form the full tensor to then recompute an approximate low-rank decomposition minizing the rank for a given epsilon global error threshold.

# Parameters
N = 50          # Full tensor is of size N x N x N
R_target = 10   # Target rank for the generated Tucker decomposition. Core tensor is of size R_target x R_target x R_target
epsilon = 1e-5  # Rank truncation global error threshold. Adjust this to see what ranks you obtain at different precision levels; 5e-1, 1e-1, 1e-2, 1e-3, ..., 1e-15

# Generate a "Decaying yet Full-Rank" Core
coords = np.ogrid[:R_target, :R_target, :R_target]
# Dominant decaying structure
G_decay = 0.25**(coords[0] + coords[1] + coords[2])
# Random "full-rank" perturbation
G_rand = np.random.randn(R_target, R_target, R_target)
# Final Core
G_true = G_decay * G_rand

# Generate orthogonal factor matrices

# Form the full tensor

# Perform ST-HOSVD with epsilon truncation
# Compute factor A

# Compute factor B

# Compute factor C and G

# Recompute full approximation [[G; A, B, C]] and display ranks and relative error of approximation
