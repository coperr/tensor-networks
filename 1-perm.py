import numpy as np

# Create the initial vector with elements 0, .., 23 using np.arange
v = np.arange(24)

# Reshape the vector into a 3x4x2 tensor
tensor = v.reshape(3, 4, 2)

# Permute tensor modes/dimensions: (3, 4, 2) -> (4, 2, 3)
# Hint: Use np.transpose(tensor) or tensor.transpose()
tensor_perm = np.transpose(tensor, (1, 2, 0))

# Print tensor shape and elements, before and after permutation
print("Original tensor shape:", tensor.shape)
print("Original tensor:")
print(tensor)

print("\n" + "-" * 40 + "\n")

print("Permuted tensor shape:", tensor_perm.shape)
print("Permuted tensor:")
print(tensor_perm)