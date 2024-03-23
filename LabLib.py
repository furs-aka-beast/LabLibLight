import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def error_of_exp(x_exp, y_exp, flag=0):
    """
    Calculating errors of linear coefficients in experiment
    :param x_exp: list -- experimental data (x-coordinate)
    :param y_exp: list -- experimental data (y-coordinate)
    :param flag: int -- if flag == 0 function will print errors of linear coefficients in experiment
    :return: er_k, er_b -- float -- errors of linear coefficients in experiment
    """
    x_exp=np.array(x_exp)
    y_exp=np.array(y_exp)
    coefficient = np.polyfit(x_exp, y_exp, 1)
    k, b = coefficient[0], coefficient[1]
    av_x = 0
    for _ in range(len(x_exp)):
        av_x += x_exp[_]
    av_x = av_x / len(x_exp)

    av_y = 0
    for _ in range(len(y_exp)):
        av_y += y_exp[_]
    av_y = av_y / len(y_exp)

    D_x = 0
    for _ in range(len(x_exp)):
        D_x += (x_exp[_] - av_x)**2
    D_x = D_x / len(x_exp)

    D_y = 0
    for _ in range(len(y_exp)):
        D_y += (y_exp[_] - av_y) ** 2
    D_y = D_y / len(y_exp)

    av_x2 = 0
    for _ in range(len(x_exp)):
        av_x2 += x_exp[_]**2
    av_x2 = av_x2 / len(x_exp)

    er_k = np.sqrt(1/(len(x_exp)-2)*((D_y/D_x)-k**2))
    er_b = er_k * np.sqrt(av_x2)
    if flag == 0:
        print('Coefficions calculeted in linear approximation:')
        print("k = ", k, "+-", er_k)
        print("b = ", b, "+-", er_b)
    if flag == 1:
        return er_k, er_b
def linear_theory(x_exp, y_exp, m = None, M = None):
    """
    :param x_exp: list -- experimental data for x-axis
    :param y_exp: list -- experimental data for y-axis
    :return: x_th, y_th -- lists with linear approximation of experimental data
    :m -- minimal x_value
    :M -- maximal x_value
    """
    if m == None: m=min(x_exp)
    if M == None: M=max(x_exp)
    exp= pd.DataFrame({"x":x_exp,"y": y_exp})
    exp=exp[exp.x<=M]
    exp=exp[exp.x>=m]
    k, b = np.polyfit(exp.x, exp.y, 1)
    x_th = np.arange(m - 0.05 * (M - m), M + 0.05 * (M - m), 0.0001 * (M - m))
    y_th = []
    for _ in range(0, len(x_th)):
        y_th.append(k * x_th[_] + b)
    y_th=np.array(y_th)
    error_of_exp(exp.x,exp.y)
    return pd.DataFrame(np.array([x_th, y_th]).transpose(), columns=['x','y'])

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
def plot_show():
    plt.legend(loc='best', fontsize=12)
    plt.show()
def const_err(x_exp, y_exp, label="Эксперементальные точки", xerr=0, yerr=0):
    plt.errorbar(x_exp, y_exp, np.full(x_exp.size, xerr), np.full(y_exp.size, yerr), fmt=".", label=label)
