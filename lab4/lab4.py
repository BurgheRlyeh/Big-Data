import numpy as np
# import pandas as pd
import scipy as sp
import scipy.linalg as scl
from dateutil import parser, rrule
from datetime import datetime, time, date
import scipy.linalg
import csv
from scipy import fft
from matplotlib import pyplot as plt

# from hurst import compute_Hc, random_walk


i = np.arange(0, 201)
h = 0.02


def model_series():
    return sum([(k * np.exp(-h * i / k) * np.cos(np.pi * (2 * k * h * i + 1 / k))) for k in range(1, 4)])


def prony(x, t):
    x = x[:len(x) - (1 if len(x) % 2 == 1 else 0)]

    p = len(x) // 2

    a = scl.solve([([0] + list(x))[p + j:j:-1] for j in range(p)], -x[p::])

    z = np.roots([*a[::-1], 1])

    h = scl.solve([z ** (n + 1) for n in range(p)], x[:p])

    f = 1 / (2 * np.pi * t) * np.arctan(np.imag(z) / np.real(z))
    alfa = 1 / t * np.log(np.abs(z))
    A = np.abs(h)
    fi = np.arctan(np.imag(h) / np.real(h))

    return f, alfa, A, fi


# def task1():


if __name__ == '__main__':
    l = model_series()
    print(l.__class__)

    print(model_series())

    # print(np.arange(0, 4) * np.arange(0, 3))
