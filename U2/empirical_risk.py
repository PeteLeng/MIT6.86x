# Calculate empirical risk of a training set
# R_n = 1/n * sum(Loss(x_i, y_i))

import numpy as np


# Use hinge loss
def emp_risk_h_loss(x_s, y_s, theta):
    loss_s = [max(0, 1-(y-np.dot(theta, x))) for x, y in zip(x_s, y_s)]
    return sum(loss_s)/len(loss_s)


# Use squared error loss
def emp_risk_se_loss(x_s, y_s, theta):
    loss_s = [(y-np.dot(theta, x))**2/2 for x, y in zip(x_s, y_s)]
    return sum(loss_s) / len(loss_s)


def main(x_s, y_s, theta):
    er_hl = emp_risk_h_loss(x_s, y_s, theta)
    print('Empirical risk using hinge loss:', er_hl)
    er_sel = emp_risk_se_loss(x_s, y_s, theta)
    print('Empirical risk using squared error loss:', er_sel)


if __name__ == '__main__':
    feature_vectors = np.array([
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, -1],
        [-1, 1, 1],
    ])
    labels = [2, 2.7, -0.7, 2]
    theta = np.array([0, 1, 2])
    main(feature_vectors, labels, theta)
