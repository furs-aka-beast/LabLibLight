import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linear_theory(x_exp, y_exp, m = None, M = None):
    """
    :param x_exp: list -- experimental data for x-axis
    :param y_exp: list -- experimental data for y-axis
    :return: x_th, y_th -- lists with linear approximation of experimental data
    :m -- minimal x_value
    :M -- maximal x_value
    """
    if m == None : m = min(x_exp)
    if M == None : M = max(x_exp)
    exp= pd.DataFrame(np.array([x_exp, y_exp]), columns=['x','y'])
    exp=exp[exp.x<M and exp.x>m]
    k, b = np.polyfit(exp.x, exp.y, 1)
    x_th = np.arange(m - 0.05 * (M - m), M + 0.05 * (M - m), 0.0001 * (M - m))
    y_th = []
    for _ in range(0, len(x_th)):
        y_th.append(k * x_th[_] + b)
    return pd.DataFrame(np.array([x_th, y_th]), columns=['x','y'])

def plot_setup(x_name, y_name):
    """
    Function for drawing plot with one curve of points (x_exp, y_exp) with linear approximation and error-bars
    :param x_name: string -- name for x-axis
    :param y_name: string -- name for x-axis
    :param legend: string -- legend for plot
    """
    plt.figure(figsize=(10, 5))
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.grid(True)
    plt.legend(loc='best', fontsize=12)
