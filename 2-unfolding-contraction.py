import numpy as np

# The goal in this exercise is to be able to do tensor unfoldings/foldings with reshape/transpose,
# then use matrix multiplications to perform the contraction on unfolded tensors on a single common dimension.

# Create a 3x4x2 tensor X with elements 0,...,23

X = np.arange(24).reshape(3, 4, 2)

print(X)

# Now compute mode-2 unfolding of the tensor X. First permute X by moving mode 1 (of size 4) to index 0

X_perm = np.transpose(X, (1, 0, 2))

# Then reshape to a matrix of size 4x6 using tensor.reshape()

X_unfold = X_perm.reshape(4, 6)

# Print the unfolded matrix dimensions and elements

print("Dimensions: ", X_unfold.shape)

print(X_unfold)

# Using this unfolding, compute the contraction X with itself along its original mode 2 (of size 4) using matrix multiplication (@ operation)

C = X_unfold.T @ X_unfold

# Then reshape the output to a tensor of dimensions 3 x 2 x 3 x 2 and print it

C = C.reshape(3,2,3,2)

print(C)

# Now define a function unfold(tensor, mode) that can unfold any tensor in any mode

def unfold(tensor, mode):
    dims = list(range(tensor.ndim))

    dims.pop(mode)
    perm = [mode] + dims

    tensor_permuted = np.transpose(tensor, perm)

    return tensor_permuted.reshape((tensor.shape[mode], -1))

# and use this function to print the unfolding of X in mode 2 to compare with the previous results

X_unfold_func = unfold(X, 1)
print(X)

# Define another function contract(tensor_x, tensor_y, mode_x, mode_y) that contracts tensor_x along mode x with tensor_y along mode y, using unfold() function and matrix multiplication

def contract(tensor_x, tensor_y, mode_x, mode_y):
    mat_x = unfold(tensor_x, mode_x)
    mat_y = unfold(tensor_y, mode_y)

    contracted_matrix = mat_x.T @ mat_y

    shape_x = list(tensor_x.shape)
    shape_x.pop(mode_x)

    shape_y = list(tensor_y.shape)
    shape_y.pop(mode_y)

    final_shape = tuple(shape_x + shape_y)

    return contracted_matrix.reshape(final_shape)

# Contract X with itself along mode 2 using contract() function, verify the result with the previous contraction (print the difference, make sure it is zero)

contraction_func = contract(X, X, 1, 1)
difference = np.sum(np.abs(C - contraction_func))
print(difference)
