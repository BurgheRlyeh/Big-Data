import math
import numpy as np
import pandas as pd
import scipy as sp
from dateutil import parser, rrule
from datetime import datetime, time, date
import scipy.linalg
import csv
from matplotlib import pyplot as plt
import cmath

ms_rng = range(1, 501)
ms_h = 0.02


# 1. Составить на языке R программу разложения ряда по методу Прони. Для
#    ее проверки сгенерировать модельный ряд
#    x_i = sum(k * exp(-h * i / k) * cos(4 * pi * k * h * i + pi / k),
#    k = 1..3, i = 1..200, h = 0.02 и применить к нему метод Прони.
def model_series():
    return np.array([sum([(
            k * np.exp(-ms_h * i / k) * np.cos(4 * np.pi * k * ms_h * i + np.pi / k)
    ) for k in range(1, 4)]) for i in ms_rng])


def prony(x: np.array, t: float):
    x = x[:len(x) - (len(x) % 2)]
    p = len(x) // 2

    z = np.roots([
        *scipy.linalg.solve([([0] + list(x))[p + i:i:-1] for i in range(p)], -x[p::])[::-1],
        1
    ])

    h = scipy.linalg.solve([z ** (n + 1) for n in range(p)], x[:p])

    return (
        np.arctan(np.imag(z) / np.real(z) / (2 * np.pi * t)),  # omega
        1 / t * np.log(np.abs(z)),  # lambda
        np.abs(h),  # A
        np.arctan(np.imag(h) / np.real(h))  # phi
    )


def task1():
    ms = model_series()

    plt.figure(num='task 1.1')
    plt.title('x_i = sum(k * exp(-h * i / k) * cos(4 * pi * k * h * i + pi / k), k = 1..3')
    plt.plot(ms)
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('x_k')

    _, _, A, _ = prony(ms, 0.1)

    plt.figure(num='task 1.2')
    plt.title('Amplitudes')
    plt.stem(A)
    plt.plot()
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('A_k')


# 2. Найти в интернете данные по среднесуточным температурам за два года
#    вашего населенного пункта и провести их анализ на наличие тренда и сезонных колебаний.
def ema(data, alpha):
    res = [data[0]]
    for i in range(1, data.size):
        res += [alpha * data[i] + (1 - alpha) * res[i - 1]]
    return res


def sma(data, m):
    res = []
    for i in range(m):
        r = range(0, 2 * i + 2)
        res += [sum(data[j] for j in r) / len(r)]
    for i in range(m, data.size - m - 1):
        r = range(i - m, i + m + 1)
        res += [sum(data[j] for j in r) / len(r)]
    for i in range(data.size - m - 1, data.size):
        r = range(2 * i - data.size, data.size)
        res += [sum(data[j] for j in r) / len(r)]
    return np.array(res)


def rotate_points(data):
    res = []
    for i in range(0, len(data) - 3):
        local = data[i: i + 3]
        if data[i] == min(local) or data[i] == max(local):
            res += [data[i]]
    return res


def kendall(data, trend):
    tail = data - trend
    n = len(tail)

    p = rotate_points(tail)
    p_c = len(p)
    print('p_c: ', p_c)

    print('Kendall coef: ', 4.0 * p_c / (n * (n - 1.0)) - 1.0)

    e = 2.0 / 3.0 * (n - 2.0)
    d = (16.0 * n - 29.0) / 90.0

    if e - d < p_c < e + d:
        print("random")
    elif e + d < p_c:
        print("rapidly oscillating")
    elif p_c < e - d:
        print("positively correlated")


def task2():
    ms = model_series()

    df = pd.read_csv('spb.csv', parse_dates=['time'], dayfirst=True)
    date_list = df['time'].tolist()
    temp_list = np.array(df['temp'].tolist())
    trend_sma_55 = sma(temp_list, 55)
    trend_ema_005 = ema(temp_list, 0.05)
    trend_ema_01 = ema(temp_list, 0.1)

    plt.figure(num='task 2.1')
    plt.title('avg daily temp')
    plt.plot(date_list, temp_list, color='blue', label='temp')
    plt.plot(date_list, trend_sma_55, label='sma, m = 55')
    plt.plot(date_list, trend_ema_005, label='ema, m =  0.05')
    plt.plot(date_list, trend_ema_01, label='ema, m = 0.1')
    plt.xlabel('date')
    plt.ylabel('temp')
    plt.grid()
    plt.legend()

    fft_orig = abs(np.fft.fft(np.array(temp_list)))

    ordi = np.linspace(0, 0.5, len(fft_orig) // 2)

    print(f"main freq = {ordi[np.argmax(fft_orig[1:len(fft_orig) // 2]) + 1]}")
    plt.figure(num='task 2.2')
    plt.title('freq')
    plt.plot(ordi[1:], fft_orig[1:len(fft_orig) // 2] / len(fft_orig), label='FFT(x)')
    plt.grid()

    print('kendall:')
    print('sma, m = 55')
    kendall(temp_list, trend_sma_55)
    print('ema, m = 0.05')
    kendall(temp_list, trend_ema_005)
    print('ema, m = 0.1')
    kendall(temp_list, trend_ema_01)


if __name__ == '__main__':
    task1()
    task2()

    plt.show()
