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
np.random.seed(42)
A_true = np.random.randn(N, R)
B_true = np.random.randn(N, R)
C_true = np.random.randn(N, R)

T = np.einsum('ir,jr,kr->ijk', A_true, B_true, C_true)
norm_T = np.linalg.norm(T)
# Now we will compute an approximate CP decomposition of T using CP-ALS.
# First, allocate and randomly initialize factor matrices that are computed during CP-ALS
A = np.random.randn(N, R)
B = np.random.randn(N, R)
C = np.random.randn(N, R)

# Perform max_iter ALS iterations and update factor matrices A, B, C at each iteration. Use pseudoinverse (np.linalg.pinv) for solving the least squares problem.
# Compute and print the relative error of approximation at each iteration.
for iteration in range(max_iter):

    T1 = T.reshape(N, N * N)
    KR_BC = np.einsum('jr,kr->jkr', B, C).reshape(N * N, R)
    A = T1 @ np.linalg.pinv(KR_BC.T)
    
    T2 = T.transpose(1, 0, 2).reshape(N, N * N)
    KR_AC = np.einsum('ir,kr->ikr', A, C).reshape(N * N, R)
    B = T2 @ np.linalg.pinv(KR_AC.T)
    
    T3 = T.transpose(2, 0, 1).reshape(N, N * N)
    KR_AB = np.einsum('ir,jr->ijr', A, B).reshape(N * N, R)
    C = T3 @ np.linalg.pinv(KR_AB.T)
    
    T_approx = np.einsum('ir,jr,kr->ijk', A, B, C)
    rel_error = np.linalg.norm(T - T_approx) / norm_T
    
    print("Iteration", iteration+1, "/", max_iter, "| Relative Error:", rel_error)
    
# Display the final relative error of approximation
print("Final Relative Error after", max_iter, "iterations:", rel_error)
