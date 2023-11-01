import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

k = np.arange(0, 201)
h = 0.1

norm = np.random.normal(0, 1, len(k))


# 1. Сгенерировать модельный ряд x_k = 0.5 * sin(k * h) + norm(0, 1), k = 0...200, h = 0.1
def model_series():
    return 0.5 * np.sin(k * h) + norm


def task1():
    ms = model_series()

    plt.figure(num='task 1')
    plt.title("x_k = 0.5 * sin(k * h) + norm(0, 1)")
    plt.plot(ms, '.', color='blue', label="model series")
    plt.legend()


# 2. Выделить тренд методом экспоненциального скользящего среднего скоэффициентами
#    alpha = 0.01, 0.05, 0.1, 0.3
def trend_series():
    return 0.5 * np.sin(k * h)


def ema(data, alpha):
    res = [data[0]]

    for i in range(1, data.size):
        res += [alpha * data[i] + (1 - alpha) * res[i - 1]]

    return res


def task2():
    ms = model_series()

    plt.figure(num='task 2')
    plt.title("ema")
    plt.plot(ms, '.', color='blue', label="series")
    plt.plot(trend_series(), color='red', label="trend")
    plt.plot(ema(ms, 0.01), label="ema, alpha = 0.01")
    plt.plot(ema(ms, 0.05), label="ema, alpha = 0.05")
    plt.plot(ema(ms, 0.1), label="ema, alpha = 0.1")
    plt.plot(ema(ms, 0.3), label="ema, alpha = 0.3")
    plt.legend()


# 4. Вычислить амплитудный спектр Фурье для этого модельного ряда и
#    определить его главную частоту
def task4():
    ms = model_series()

    plt.figure(num='task 4')
    plt.title('Fourier amplitude spectrum')
    amp_spec = 2 / len(k) * np.abs(np.fft.fft(ms)[:len(ms) // 2])
    freqs = np.linspace(0, 0.5, len(ms) // 2)
    plt.plot(freqs, amp_spec)

    print('frequency = ', freqs[np.argmax(amp_spec)])
    print()


# 5. Вычесть тренды из ряда и проверить остатки на случайность,
#    несмещенность их средних и нормальность.
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

    print('kendall coef: ', 4.0 * p_c / (n * (n - 1.0)) - 1.0)

    e = 2.0 / 3.0 * (n - 2.0)
    d = (16.0 * n - 29.0) / 90.0

    if e - d < p_c < e + d:
        print("random")
    elif e + d < p_c:
        print("rapidly oscillating")
    elif p_c < e - d:
        print("positively correlated")

    print('mean = ', tail.mean())
    print('std deviation = ', st.tstd(tail))
    print('normality = ', st.normaltest(tail)[1])
    print()


def task5():
    ms = model_series()

    print("ema, m = 0.01")
    kendall(ms, ema(ms, 0.01))
    print("ema, m = 0.05")
    kendall(ms, ema(ms, 0.05))
    print("ema, m = 0.1")
    kendall(ms, ema(ms, 0.1))
    print("ema, m = 0.3")
    kendall(ms, ema(ms, 0.3))


if __name__ == '__main__':
    task1()
    task2()
    task4()
    task5()

    plt.show()
