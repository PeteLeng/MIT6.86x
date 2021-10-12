# Collaborative filtering
# Assume the projected matrix has low rank X(a x i) = np.matmul(U(axd), V(dxi))
# The objective is to minimize
# j = sum( (y_ai - sum_over_d(u_ad * v_di) )**2/2
#     + lambda/2 * ( sum_over_u(u_ai**2) + sum_over_v(v_ai**2) )
import numpy as np
import sympy as sp

# - initialize v
# - calculate uv
# - for each row in uv, calculate the squared error and the regularization term
# - differentiate, solve for 0


def iterative_factorization(observed_matrix, reg_coeff, num_iterations, **kwargs):
    factor, multiple = factorization(observed_matrix, reg_coeff, **kwargs)
    # print('Given factor:\n', multiple, '\n', 'Estimated:\n', factor)
    for i in range(num_iterations):
        factor, multiple = factorization(observed_matrix, reg_coeff, factor_matrix=factor)
        # print('Given factor:\n', multiple, '\n', 'Estimated:\n', factor)
    if multiple.shape[1] == factor.shape[0]:
        return np.matmul(multiple, factor)
    else:
        return np.matmul(factor, multiple)


def factorization(observed_matrix, reg_coeff, factor_matrix=None, rank=None, direction=None):
    """
    Given a observed matrix, its assumed rank, an initialed factor (optional),
    find the multiple that optimized the objective function,
    consisting of squared errors and regularization term.
    :param observed_matrix: np.matrix
        The user-product matrix
    :param reg_coeff: int
        Regularization terms
    :param factor_matrix: np.matrix
        Initialized factor matrix
    :param rank: int
        Assumed rank of the user-product matrix
    :param direction: bool
        Decide which matrix to initialize, 1 for product matrix, 0 for user matrix
    :return: tuple of np.matrix

    """
    n_row, n_col = observed_matrix.shape
    if factor_matrix is not None:
        factor_row, factor_col = factor_matrix.shape
        if factor_row in (n_row, n_col):
            direction = 0
            multiple_matrix = symbolic_matrix([factor_col, n_col])
        elif factor_col in (n_row, n_col):
            direction = 1
            multiple_matrix = symbolic_matrix([n_row, factor_row])
        else:
            raise ValueError('Dimension of factor matrix does not fit observed matrix')
    elif rank is not None and direction is not None:
        if direction:
            factor_matrix = np.ones([rank, n_col])
            multiple_matrix = symbolic_matrix([n_row, rank])
        else:
            factor_matrix = np.ones([n_row, rank])
            multiple_matrix = symbolic_matrix([rank, n_col])
    else:
        raise ValueError('Provide either factor_matrix or rank and direction')

    if direction:
        est_matrix = np.matmul(multiple_matrix, factor_matrix)
    else:
        est_matrix = np.matmul(factor_matrix, multiple_matrix)
    observed = ~np.isnan(observed_matrix)
    squared_errors = np.sum((est_matrix[observed] - observed_matrix[observed])**2)/2
    regularization = np.sum(multiple_matrix**2) * reg_coeff/2
    obj = squared_errors + regularization
    est_multiple_matrix = np.empty_like(multiple_matrix).astype('object')
    for entry in multiple_matrix.flat:
        i, j = int(entry.name[-2]), int(entry.name[-1])
        gradient = sp.diff(obj, entry)
        est_multiple_matrix[i, j] = sp.solveset(gradient, entry).args[0]

    return est_multiple_matrix, factor_matrix


def symbolic_matrix(dimension, name='m'):
    """
    Given the dimension, populate a matrix of sympy variables
    :param dimension: list
        list of rows and columns
    :param name: str
        name of the matrix
    :return:
    """
    [n_row, n_col] = dimension
    m = np.empty([n_row, n_col]).astype('object')
    # m[:] = np.NaN
    for row in range(n_row):
        for col in range(n_col):
            entry = f'{name.lower()}_{row}{col}'
            m[row, col] = sp.Symbol(entry)
    return m


if __name__ == '__main__':
    v = np.array([[4, 2, 1]])
    examples = np.array([
        [1, 8, np.NaN],
        [2, np.NaN, 5]
    ])
    lmda = sp.Symbol('lmda')

    # Test symbolic matrix
    # ar1 = symbolic_matrix(3, 3, 'u')
    # print(ar1)

    # Test factorization
    res = factorization(examples, 1, factor_matrix=v)
    # print(res)

    # Test iterative factorization
    res = iterative_factorization(examples, 1, 5, factor_matrix=v)
    print(res)

    # Debug
    # product = np.array([[0.637439110697700, 3.05324815909688, 1.99458569294239]])
    # print(factorization(examples, 1, factor_matrix=product))