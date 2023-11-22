import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def huber(x: np.array, k: float):
    return np.mean([elem if abs(elem) < k else k * np.sign(elem) for elem in x])


def boxplot_rule(x: np.array):
    return [i for i in x if i not in plt.boxplot(x)["fliers"]]


def double_stage_mean(x: np.array):
    return np.mean(huber(boxplot_rule(x), 1.44))


distributions = {
    "norm": stats.norm.rvs,
    "cauchy": stats.cauchy.rvs,
    "mix": lambda size: 0.9 * stats.norm.rvs(size=size) + 0.1 * stats.cauchy.rvs(size=size)
}

measures = {
    "mean": np.mean,
    "median": np.median,
    "huber": lambda x: huber(x, 1.44),
    "double_stage": double_stage_mean
}


def monte_karlo(n: int, sample_size: int, dist_grvs, measure):
    means = [measure(dist_grvs(size=sample_size)) for _ in range(n)]

    return np.mean(means), np.var(means)


def task():
    for dist, f_dist in distributions.items():
        print(dist)
        for measure, f_measure in measures.items():
            mu, var = monte_karlo(10000, 100, f_dist, f_measure)
            print(f"\t{measure}")
            print(f"\t\tMean:\t{mu:.6f}")
            print(f"\t\tDisp:\t{var:.6f}")
        print("")


if __name__ == "__main__":
    task()
