import numpy as np

# The goal of this exercise is to first create a random rank-R CP tensor/decomposition, then try to rediscover this decomposition using CP-ALS on the full rank-R tensor.

# Problem parameters:
# factor matrices A_true, B_true, C_true are of size N x R,
# tensor T = [[A_true, B_true, C_true]] is of size N x N x N,
# CP-ALS should run for max_iter iterations.
N = 50
R = 5
max_iter = 50

# Generate Ground Truth; create random factor matrices A_true, B_true, C_true of size N x R that form a CP decomposition, then form the corresponding full tensor T using einsum

# Now we will compute an approximate CP decomposition of T using CP-ALS.
# First, allocate and randomly initialize factor matrices that are computed during CP-ALS

# Perform max_iter ALS iterations and update factor matrices A, B, C at each iteration. Use pseudoinverse (np.linalg.pinv) for solving the least squares problem.
# Compute and print the relative error of approximation at each iteration.

# Display the final relative error of approximation
