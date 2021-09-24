# An interactive display of perceptron update

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)


class Perceptron(tk.Tk):
    def __init__(self, training_set, offset=False):
        super().__init__('The perceptron')
        self.examples = training_set
        self.offset = offset
        self.create_frames()
        self.create_widgets()
        self.theta = np.zeros(2)
        self.theta0 = 0
        self.pointer = 0
        self.btn_clear.bind('<Button-1>', self.clear_fig)
        self.btn_next.bind('<Button-1>', self.perceptron)

    def create_frames(self):
        self.frm_control = tk.Frame(self)
        self.frm_display = tk.Frame(self)

        # Frame layout
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self.frm_control.grid(row=0, column=0, sticky='ew')
        self.frm_display.grid(row=1, column=0, sticky='nsew')

    def create_widgets(self):
        self.btn_next = tk.Button(self.frm_control, text='Next')
        self.btn_prev = tk.Button(self.frm_control, text='Prev')
        self.btn_clear = tk.Button(self.frm_control, text='Clear')
        self.txt_out = tk.Text(self.frm_display, width=30, bg='linen')
        self.plot_examples()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_display)
        # self.canvas.draw()

        # Widgets layout
        self.btn_next.grid(row=0, column=0)
        self.btn_prev.grid(row=0, column=1)
        self.btn_clear.grid(row=0, column=2)
        self.frm_display.rowconfigure(0, weight=1)
        self.frm_display.columnconfigure(1, weight=1)
        self.txt_out.grid(row=0, column=0, sticky='ns')
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky='nsew')

    def plot_examples(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot()
        self.ax.axline((0, 0), (1, 0))
        self.ax.axline((0, 0), (0, 1))
        def color(x): return 'red' if x<0 else 'green'
        for exp in self.examples:
            data = exp[0]
            self.ax.scatter(data[0], data[1], color=color(exp[1]))
        # plt.show()

    def clear_fig(self, event):
        # Deactivate the canvas widget
        # self.canvas.get_tk_widget().grid_forget()

        # Redraw the fig and canvas
        # self.fig = Figure()
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.frm_display)
        # self.canvas.get_tk_widget().grid(row=0, column=0, stick='nsew')

        # Redraw ax
        self.fig.clear()
        self.canvas.draw()

    def plot_boudary(self):
        theta1, theta2 = self.theta
        theta0 = self.theta0
        if len(self.ax.lines) > 2:
            self.ax.lines.pop()
            self.arrow.remove()
        if self.offset:
            if theta2 == 0:
                self.ax.axline((-theta0/theta1, 0), (-theta0/theta1), c='black', ls='--')
            else:
                self.ax.axline((0, -theta0 / theta2), (1, -(theta0 + theta1) / theta2), c='black', ls='--')
        else:
            if theta2 == 0:
                self.ax.axline((0, 0), (0, 1), c='black', ls='--')
            else:
                self.ax.axline((0, 0), (1, -theta1/theta2), c='black', ls='--')
        self.arrow = self.ax.arrow(0, 0, theta1, theta2)
        self.canvas.draw()

    def update_theta(self):
        x, y = self.examples[self.pointer]
        if self.offset:
            if not y * (np.matmul(self.theta.transpose(), np.array(x)) + self.theta0) > 0:
                self.theta += y * np.array(x)
                self.theta0 += y
                print('update to {}, {}'.format(self.theta, self.theta0))
        else:
            if not y * (np.matmul(self.theta.transpose(), np.array(x))) > 0:
                self.theta += y * np.array(x)
                print('update to {}'.format(self.theta))

    def perceptron(self, event):
        print('testing example {}'.format(self.examples[self.pointer][0]))
        self.update_theta()
        if not self.theta.all() == 0:
            self.plot_boudary()
        self.pointer = (self.pointer + 1) % len(self.examples)

if __name__ == '__main__':
    # training_set = [([-4, 2], 1), ([-2, 1], 1), ([-1, -1], -1), ([2, 2], -1), ([1, -2], -1)]
    # homework3 = [([-1, 1], 1), ([1, -1], 1), ([1, 1], -1), ([2, 2], -1)]
    homework6 = [([-1, 0], 1), ([0, 1], 1)]
    homework6_reverse = [([0, 1], 1), ([-1, 0], 1)]
    bot = Perceptron(homework6_reverse)
    bot.mainloop()