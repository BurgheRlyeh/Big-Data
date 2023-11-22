import numpy as np
from matplotlib import pyplot as plt

norm = np.random.standard_normal(195)


def model_series():
    ms = np.append(norm, [5, -4, 3.3, 2.99, -3])
    ms.sort()
    return ms


def task1():
    ms = model_series()

    outs = []
    out_ids = []
    norm_ids = []

    for i, val in enumerate(ms):
        if np.abs(val - np.mean(ms)) > 3 * np.var(ms):
            outs += [val]
            out_ids += [i]
        else:
            norm_ids += [i]

    print('outliers: ', outs)

    plt.figure(num='task 1')
    plt.title('3 sigma outliers')
    plt.plot(norm_ids, np.delete(ms, out_ids), '.', label='x')
    plt.plot(out_ids, outs, 'o', color='red', label='outliers')
    plt.grid()
    plt.legend()


def task2():
    ms = model_series()

    plt.figure(num='task 2')
    plt.title("Tukey's boxplot")
    print("Tukey's outliers: ", [x.get_xdata() for x in plt.boxplot(ms, vert=False)["fliers"]])


if __name__ == "__main__":
    task1()
    task2()

    plt.show()
