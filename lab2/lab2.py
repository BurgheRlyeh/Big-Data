import numpy as np
import matplotlib.pyplot as plt

k = np.arange(0, 201)
h = 0.05

norm = np.random.normal(0, 1, len(k))


def model_series():
    return np.sqrt(k * h) + norm


def task1():
    ms = model_series()

    plt.figure(num='task 1')
    plt.title("x_k = sqrt(k * h) + norm(0, 1)")
    plt.plot(ms, '.', color='blue', label="model series", )
    plt.legend()
    # plt.show()


def trend_series():
    return np.sqrt(k * h)


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


def task2():
    ms = model_series()

    plt.figure(num='task 2')
    plt.title("simple moving average")
    plt.plot(ms, '.', color='blue', label="model series")
    plt.plot(trend_series(), color='red', label="trend")
    plt.plot(sma(ms, 10), label="sma, m = 10")
    plt.plot(sma(ms, 25), label="sma, m = 25")
    plt.plot(sma(ms, 55), label="sma, m = 55")
    plt.legend()
    # plt.show()


def mm(data, m):
    res = []

    for i in range(m):
        res += [np.median(data[0:2 * i + 1])]

    for i in range(m, data.size - m - 1):
        res += [np.median(data[i - m:i + m])]

    for i in range(data.size - m - 1, data.size):
        res += [np.median(data[2 * i - data.size:data.size])]

    return np.array(res)


def task3():
    ms = model_series()

    plt.figure(num='task 3')
    plt.title("moving median")
    plt.plot(ms, '.', color='blue', label="series")
    plt.plot(trend_series(), color='red', label="trend")
    plt.plot(mm(ms, 10), label="mm, m = 10")
    plt.plot(mm(ms, 25), label="mm, m = 25")
    plt.plot(mm(ms, 55), label="mm, m = 55")

    plt.legend()
    # plt.show()


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


def task4():
    ms = model_series()

    print("sma, m = 10")
    kendall(ms, sma(ms, 10))
    print()

    print("sma, m = 25")
    kendall(ms, sma(ms, 25))
    print()

    print("sma, m = 55")
    kendall(ms, sma(ms, 55))
    print()

    print("mm, m = 10")
    kendall(ms, mm(ms, 10))
    print()

    print("mm, m = 25")
    kendall(ms, mm(ms, 25))
    print()

    print("mm, m = 55")
    kendall(ms, mm(ms, 55))
    print()


if __name__ == '__main__':
    task1()
    task2()
    task3()
    task4()

    plt.show()
