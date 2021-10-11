import numpy as np
import math


# Kernel functions, K(x_1, x_2) = < phi(x_1), phi(x_2) >, an inner product of two feature mappings

# Radial base kernel
# K(x_1, x_2) = exp( -1/2 * ||x_1 - x_2||**2 )

def radial_kernel(x_1, x_2):
    return math.exp(-1/2 * np.dot(x_1-x_2, x_1-x_2))


if __name__ == '__main__':
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    print(radial_kernel(a, b))
