import numpy as np

n = 2
N = 10

xes = np.random.random_sample((N, n))  # N массивов по n в длину
yes = np.random.random_sample((N,))  # Один длинный массив длины N


print(xes)
print("xes")
print(yes)
print("yes")
y_mean = np.sum(yes) / N  # средний от всех y <y>

# x1_mean = np.sum(xes.transpose()[0]) / N
# x2_mean = np.sum(xes.transpose()[1]) / N

# 1 относится к тому, что массив получится одномерный
x_1_means = np.array(
    [sum(xes.transpose()[i]) / N for i in range(n)])  # считаем среднее от всех координат x по отдельности. <x_i>

xy_1_means = np.array(
    [np.sum(yes * xes.transpose()[i_s]) / N for i_s in range(n)]) # <yx_i>

# 2 относится к тому, что массив получится двумерный. Получается матричка размера n на n.
x_2_means = np.array([np.array([
    np.sum(xes.transpose()[i] * xes.transpose()[i_s]) / N
    for i in range(n)]) for i_s in range(n)]) # <x_ix_is>


def matrix_function(i, i_s):
    if i_s == 0:
        # мы попали в самое первое уравнение
        if i == 0:
            return 1
        else:
            return x_1_means[i - 1]
    else:
        # мы попали в более сложный случай
        if i == 0:
            return x_1_means[i_s - 1]
        else:
            return x_2_means[i_s - 1][i - 1]


def right_part_function(i_s):
    if i_s == 0:
        return y_mean
    else:
        return xy_1_means[i_s - 1]

# n+1 поскольку у нас 1ое уравнение есть и все остальные, которых n

# для нахождения a_0 и a_i нужно будет обращать main_matrix
main_matrix = np.array([np.array([matrix_function(i, i_s) for i in range(n + 1)]) for i_s in range(n + 1)])
# напишем же правую часть уравнения
right_part = np.array([right_part_function(i_s) for i_s in range(n + 1)])
# Это строчка вида [a_0, a_1, a_2, ...]
our_result = np.linalg.solve(main_matrix, right_part)
print(our_result)