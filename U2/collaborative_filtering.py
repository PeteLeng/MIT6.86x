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


def collab_filter(n_row, n_col, rank, reg_coeff, real_y, fix_m, fix='col'):
    if fix == 'col':
        row_matrix = symbolic_matrix(n_row, rank)
        col_matrix = fix_m
        proj_y = np.matmul(row_matrix, col_matrix)
        observed = ~np.isnan(real_y)
        squared_errors = np.sum((proj_y[observed] - real_y[observed])**2)/2
        regularization = np.sum(row_matrix**2) * reg_coeff/2
        obj = squared_errors + regularization
        best_row_matrix = np.empty([n_row, rank]).astype('object')
        print(best_row_matrix)
        for entry in row_matrix.flat:
            i, j = int(entry.name[-2]), int(entry.name[-1])
            print(i, j)
            gradient = sp.diff(obj, entry)
            best_row_matrix[i, j] = sp.solveset(gradient, entry)

    # else:
    #     row_matrix = np.ones(n_row, rank)
    #     col_matrix = symbolic_matrix([rank, n_col])

    return best_row_matrix


def symbolic_matrix(n_row, n_col, name='m'):
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
    print(collab_filter(2, 3, 1, lmda, examples, v))

    # Test symbolic matrix
    # ar1 = symbolic_matrix(3, 3, 'u')
    # print(ar1)
