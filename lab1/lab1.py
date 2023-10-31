import math
import numpy as np
import matplotlib.pyplot as plt


# 1. Построить на одном графике функции y=sin(x) и y=cos(x)
def task1():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

    plt.figure(figsize=(10, 5))

    plt.title('sin(x) & cos(x)')

    plt.plot(x, np.sin(x), label='y = sin(x)', color='orange')
    plt.plot(x, np.cos(x), label='y = cos(x)', color='blue')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.legend()
    plt.show()


# 2. Создать два вектора: x=-10,-9,…,5; y=-5,-4,…,10
def task2():
    x = np.arange(-10, 6)
    y = np.arange(-5, 11)

    print('x: ', x)
    print('y: ', y)

    return x, y


# 3. Построить новый вектор z, в котором на нечетных местах будут стоять элементы
# вектора x, а на четных – элементы вектора y. Отсортировать элементы
# полученного вектора.
def task3(x, y):
    z = np.empty(x.size + y.size)
    z[0::2] = x
    z[1::2] = y

    print('z: ', z)

    z.sort()
    print('sorted z: ', z)

    return z


# 4. Написать функцию, вычисляющую норму вектора, и найти нормы векторов x,y,z.
def norm(vector):
    return math.sqrt(sum(e ** 2 for e in vector))


def task4(x, y, z):
    print('x norm: ', norm(x))
    print('y norm: ', norm(y))
    print('z norm: ', norm(z))


# 5. Создать матрицу А, содержащую 10 строк и 10 столбцов, заполненную
# элементами от 1 до 100. Создать векторы, состоящие из суммы элементов
# матрицы по столбцам и по строкам. Найти произведение исходной матрицы на
# каждый из полученных векторов. Из исходной матрицы А получить матрицу В,
# исключив последние 5 строк и последние 5 столбцов.
def task5():
    matrix = np.arange(1, 101).reshape(10, 10)
    print('matrix: ', matrix)

    sum_cols = np.sum(matrix, axis=0)
    print('sums of cols: ', sum_cols)

    sum_rows = np.sum(matrix, axis=1)
    print('sums of rows: ', sum_rows)

    print('matrix * sums of cols: ', matrix.dot(sum_cols))
    print('matrix * sums of rows: ', matrix.dot(sum_rows))

    print('submatrix: ', matrix[5:10, 5:10])


# 6. Написать функцию, вычисляющую факториал числа.
def fact(n):
    return 1 if n <= 1 else n * fact(n - 1)


def task6(n):
    print('factorial(', n, ') = ', fact(n))


# 7. Создайте вектор, состоящий из 5 элементов, с помощью ввода с клавиатуры.
# Найдите минимальный и максимальный элементы вектора, а также сумму
# элементов вектора.
def task7():
    vector = [int(n) for n in input().split()]
    print('min: ', min(vector))
    print('max: ', max(vector))
    print('sum: ', sum(vector))


if __name__ == '__main__':
    task1()
    x, y = task2()
    z = task3(x, y)
    task4(x, y, z)
    task5()
    task6(np.random.randint(1, 10))
    task7()
