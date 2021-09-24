# The perceptron algorithm
# Given a set of examples in tuples, the first element as the vector, the second element as the label

import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
def perceptron(examples, passes):
    theta = None
    for iter in range(passes):
        is_updated = 0
        for d in examples:
            if theta is None:
                length = len(d[0])
                theta = np.zeros(length)
            if not d[1] * np.matmul(theta, np.array(d[0])) > 0:
                is_updated = 1
                previous = theta
                theta = theta + d[1] * np.array(d[0])
                print('update theta from {} to {}'.format(previous, theta))
        if not is_updated:
            print('training set is linearly separable')
            return theta
    print('not converged')

def perceptron_general(examples, passes):
    theta = None
    theta0 = 0
    for iter in range(passes):
        is_updated = 0
        for d in examples:
            if theta is None:
                length = len(d[0])
                theta = np.zeros(length)
            if not d[1] * np.matmul(theta, np.array(d[0])) + theta0 > 0:
                is_updated = 1
                previous = theta, theta0
                theta = theta + d[1] * np.array(d[0])
                theta0 = theta0 + d[1]
                print('update theta from {} to {}'.format(previous[0], theta))
                print('update theta from {} to {}'.format(previous[1], theta0))
        if not is_updated:
            print('training set is linearly separable')
            return theta
    print('not converged')
    
def test():
    examples = [([-1, -1], 1), ([1, 0], -1), ([-1, 10], 1)]
    perceptron(examples, 10)
    perceptron_general(examples, 10)
    
if __name__ == '__main__':
    test()
    print(np.array([-1,-1]))
