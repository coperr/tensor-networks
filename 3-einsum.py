import numpy as np
import math

# The goal of this exercise is to perform various types of einsum operations using vectors, matrices, tensors, or a combination of them.

# Initialize and display vectors, matrices, tensors
v = np.arange(4)
w = np.arange(4, 8)
M = np.arange(12).reshape(3, 4)
N = np.arange(12).reshape(4, 3)
T = np.arange(24).reshape(2, 3, 4)
print(v)
print(w)
print(M)
print(N)
print(T)

# Compute and print sum of all elements in v, M, T

# Compute and print Frobenius norm of v, M, T

# Transpose M and T by reverting the order of dimensions

# Extract the diagonal of M into a vector of size 3, keeping in mind that M is 3x4

# Compute the trace (sum of the diagonal) of M

# Expand v into a matrix whose diagonal is v and offdiagonal elements are zero

# Scalar product of v with w

# Outer product of v with w

# Tensor product of v with M to form a 3D tensor

# Matrix multiplication M * N

# Hadamard (element-wise) product of v with w and M with N'

# Batch matrix multiplication using T twice, with first dimension (size 2) as batch dimension
# Note: T is (2, 3, 4). For T*T to work as batch MM, inner dims must match.
# Here we multiply T by its own transpose in the last two dims: (2, 3, 4) x (2, 4, 3), so there are two 3 x 4 x 3 matrix multiplications to perform in batch

# Tensor-matrix multiplication of T with M along mode 1 and N along mode 2
# Mode 1 (dim 1): T(2,3,4) with M(3,4) -> result(2,4,4)
# Mode 2 (dim 2): result(2,4,4) with N(4,3) -> final(2,4,3)

# Khatri-rao product of M and N' (column-wise Kronecker)

# Kronecker product of M with N

# Tensor product of M with N
