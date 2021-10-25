# Given a set of examples,
# apply a layer of neural network,
# plot the result in terms of the hidden unit activations and see if it's linearly separable

import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from hyperbolic import tanh

# plot the input layer
# process the hidden output
# plot the output layer
class HiddenLayer(tk.Tk):
    def __init__(self, examples=None, labels=None):
        super().__init__('Neural Net')
        self.examples = examples
        self.labels = labels
        self.para_set = None
        self.idx = 0
        self.act_f = None
        self.transformed = None
        self.fig = Figure()
        self.create_frames()
        self.create_widgets()
        self.btn_input.bind('<Button-1>', self.plot_input)
        self.btn_hidden.bind('<Button-1>', self.plot_hidden)
        self.btn_next.bind('<Button-1>', self.next)

    def create_frames(self):
        self.frm_control = tk.Frame(self)
        self.frm_display = tk.Frame(self)

        # Frame layout
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.frm_control.grid(row=0, column=0, sticky='ew')
        self.frm_display.grid(row=1, column=0, sticky='nsew')

    def create_widgets(self):
        self.btn_input = tk.Button(self.frm_control, text='Input')
        self.btn_hidden = tk.Button(self.frm_control, text='Hidden')
        self.btn_next = tk.Button(self.frm_control, text='Next')
        # self.plot(self.examples, self.labels)
        self.txt_out = tk.Text(self.frm_display, width=20, bg='lightgrey')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_display)

        # Widgets layout
        self.btn_input.grid(row=0, column=0)
        self.btn_hidden.grid(row=0, column=1)
        self.btn_next.grid(row=0, column=2)
        self.frm_display.rowconfigure(0, weight=1)
        self.frm_display.columnconfigure(1, weight=1)
        self.txt_out.grid(row=0, column=0, sticky='ns')
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky='nsew')

    def set_para_set(self, para_set):
        self.para_set = para_set

    def update_para(self):
        """
        weights: a kxd matrix,
        k is the number of hidden units,
        d is the number of input nodes (dimension of the feature vector)
        each row is the vector of weights for a specific hidden unit
        offsets: a kx1 vector
        """
        self.weights = self.para_set[self.idx][0]
        self.offsets = self.para_set[self.idx][1]

    def transform(self):
        """
        :return: the activation matrix of nxk,
        X @ W' + K, applied the activation function
        """
        weighted_input = (self.examples @ self.weights.T
                          + np.vstack([self.offsets]*self.examples.shape[0]))
        if self.act_f is None:
            def f(x): return x
            return f(weighted_input)
        else:
            return self.act_f(weighted_input)

    def plot(self, examples, labels):
        # self.fig = Figure()
        self.fig.clear()
        self.ax = self.fig.add_subplot()
        self.ax.axline((0, 0), (1, 0))
        self.ax.axline((0, 0), (0, 1))
        def color(x): return 'red' if x < 0 else 'blue'
        for f, l in zip(examples, labels):
            self.ax.scatter(f[0], f[1], color=color(l))
        self.canvas.draw()

    def display_para(self):
        self.txt_out.delete('1.0', tk.END)
        self.txt_out.insert('1.0', self.weights)

    def plot_input(self, event):
        self.plot(self.examples, self.labels)

    def plot_hidden(self, event):
        self.transformed = self.transform()
        # print(self.hidden)
        self.plot(self.transformed, self.labels)

    def next(self, event):
        self.idx = (self.idx + 1) % len(self.para_set)
        self.update_para()
        self.fig.clear()
        self.canvas.draw()
        self.display_para()


def relu(x):
    return x * (x > 0)


def main():
    examples = np.array([
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1]
    ])
    labels = np.array([1, -1, -1, 1])
    para_set = [
        [
            np.array([
                [0, 0],
                [0, 0],
            ]),
            np.array([0, 0]),
        ],
        [
            np.array([
                [2, 2],
                [-2, -2],
            ]),
            np.array([1, 1]),
        ],
        [
            np.array([
                [-2, -2],
                [2, 2],
            ]),
            np.array([1, 1]),
        ]
    ]
    neuron = HiddenLayer(examples, labels)
    neuron.set_para_set(para_set)
    neuron.act_f = relu
    neuron.mainloop()


if __name__ == "__main__":
    main()
