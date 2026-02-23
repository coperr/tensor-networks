import numpy as np

# The goal in this exercise is to be able to do tensor unfoldings/foldings with reshape/transpose,
# then use matrix multiplications to perform the contraction on unfolded tensors on a single common dimension.

# Create a 3x4x2 tensor X with elements 0,...,23

# Now compute mode-2 unfolding of the tensor X. First permute X by moving mode 1 (of size 4) to index 0

# Then reshape to a matrix of size 4x6 using tensor.reshape()

# Print the unfolded matrix dimensions and elements

# Using this unfolding, compute the contraction X with itself along its original mode 2 (of size 4) using matrix multiplication (@ operation)

# Then reshape the output to a tensor of dimensions 3 x 2 x 3 x 2 and print it

# Now define a function unfold(tensor, mode) that can unfold any tensor in any mode

# and use this function to print the unfolding of X in mode 2 to compare with the previous results

# Define another function contract(tensor_x, tensor_y, mode_x, mode_y) that contracts tensor_x along mode x with tensor_y along mode y, using unfold() function and matrix multiplication

# Contract X with itself along mode 2 using contract() function, verify the result with the previous contraction (print the difference, make sure it is zero)
