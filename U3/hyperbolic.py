import matplotlib.pyplot as plt
import numpy as np


def cosh(x): return (np.exp(x) + np.exp(-x)) / 2


def sinh(x): return (np.exp(x) - np.exp(-x)) / 2


def tanh(x): return sinh(x) / cosh(x)


def plot_function(function, start, end):
    x_array = np.linspace(start, end)
    y_array = [function(x) for x in x_array]
    plt.plot(x_array, y_array, label='plot of {} for range ({}, {})'.format(function.__name__, start, end))
    plt.legend()


def run(start, end):
    plot_function(cosh, start, end)
    plot_function(sinh, start, end)
    plot_function(tanh, start, end)


if __name__ == "__main__":
    fig = plt.figure()
    run(-3, 3)