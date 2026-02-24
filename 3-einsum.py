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
sum_v = np.einsum('i->', v)
sum_M = np.einsum('ij->', M)
sum_T = np.einsum('ijk->', T)
print("Sum v: ", sum_v," M: ", sum_M," T: ",sum_T)

# Compute and print Frobenius norm of v, M, T
norm_v = math.sqrt(np.einsum('i,i->', v, v))
norm_M = math.sqrt(np.einsum('ij,ij->', M, M))
norm_T = math.sqrt(np.einsum('ijk,ijk->', T, T))
print("Frobenius norm v: ", norm_v, "M: ",norm_M," T: ", norm_T)

# Transpose M and T by reverting the order of dimensions
M_T = np.einsum('ij->ji', M)
T_T = np.einsum('ijk->kji', T)
print("\nTransposed M:\n", M_T)

# Extract the diagonal of M into a vector of size 3, keeping in mind that M is 3x4
print("\nDiagonal of M:", np.einsum('ii->i', M[:, :3]))

# Compute the trace (sum of the diagonal) of M
print("Trace of M:", np.einsum('ii->', M[:, :3]))

# Expand v into a matrix whose diagonal is v and offdiagonal elements are zero
print("\nDiagonal matrix from v:\n", np.einsum('i,ij->ij', v, np.eye(4)))

# Scalar product of v with w
print("\nScalar product v.w:", np.einsum('i,i->', v, w))

# Outer product of v with w
print("\nOuter product v and w:\n", np.einsum('i,j->ij', v, w))

# Tensor product of v with M to form a 3D tensor
print("\nTensor product v tensor M:\n", np.einsum('i,jk->ijk', v, M))

# Matrix multiplication M * N
print("\nMatrix multiplication M*N:\n", np.einsum('ij,jk->ik', M, N))

# Hadamard (element-wise) product of v with w and M with N'
print("\nHadamard v*w:", np.einsum('i,i->i', v, w))
print("Hadamard M*N':\n", np.einsum('ij,ij->ij', M, N.T))

# Batch matrix multiplication using T twice, with first dimension (size 2) as batch dimension
# Note: T is (2, 3, 4). For T*T to work as batch MM, inner dims must match.
# Here we multiply T by its own transpose in the last two dims: (2, 3, 4) x (2, 4, 3), so there are two 3 x 4 x 3 matrix multiplications to perform in batch
print("\nBatch MM T x T^T:\n", np.einsum('bij,bjk->bik', T, T.transpose(0, 2, 1)))

# Tensor-matrix multiplication of T with M along mode 1 and N along mode 2
# Mode 1 (dim 1): T(2,3,4) with M(3,4) -> result(2,4,4)
res_mode1 = np.einsum('bik,ij->bjk', T, M)

# Mode 2 (dim 2): result(2,4,4) with N(4,3) -> final(2,4,3)
print("\nFinal Tensor Matrix multiplication shape:", np.einsum('bjk,kl->bjl', res_mode1, N).shape)

# Khatri-rao product of M and N' (column-wise Kronecker)
khatri_rao = np.einsum('ik,jk->ijk', M, N.T).reshape(-1, M.shape[1])
print("\nKhatri-Rao M and N' shape:", khatri_rao.shape)

# Kronecker product of M with N
kronecker_MN = np.einsum('ij,kl->ikjl', M, N).reshape(M.shape[0]*N.shape[0], M.shape[1]*N.shape[1])
print("Kronecker M and N shape:", kronecker_MN.shape)

# Tensor product of M with N
tensor_MN = np.einsum('ij,kl->ijkl', M, N)
print("Tensor product M and N shape:", tensor_MN.shape)
