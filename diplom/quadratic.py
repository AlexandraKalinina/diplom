import numpy as np

n = 2
N = 10

xes = np.random.random_sample((N, n))
xt = xes.transpose()
print(xes)
print("xes")
# yes = np.random.random_sample((N,))
yes = 1 + np.dot(np.ones(n), xt) + np.dot(np.ones(n), xt**2)
print("ones")
print(np.ones(n))
print("yes")
print(yes)
# 0 мерные
y_mean = np.sum(yes) / N
# 1 мерные
x_mean = np.sum(xes, axis=0) / N
xy_mean = np.array([np.sum(
    xt[i] * yes
) / N for i in range(n)])
# 2 мерные
xx_mean = np.array([np.array([np.sum(
    xt[i] * xt[j]
) / N for j in range(n)]) for i in range(n)])
xxy_mean = np.array([np.array([np.sum(
    xt[i] * xt[j] * yes
) / N for j in range(n)]) for i in range(n)])
# 3 мерные
xxx_mean = np.array([np.array([np.array([np.sum(
    xt[i] * xt[j] * xt[i_s]
) / N for i_s in range(n)]) for j in range(n)]) for i in range(n)])
xxxy_mean = np.array([np.array([np.array([np.sum(
    xt[i] * xt[j] * xt[i_s] * yes
) / N for i_s in range(n)]) for j in range(n)]) for i in range(n)])
# 4 мерный
xxxx_mean = np.array([np.array([np.array([np.array([np.sum(
    xt[i] * xt[j] * xt[i_s]
) / N for j_s in range(n)]) for i_s in range(n)]) for j in range(n)]) for i in range(n)])

num_to_coef = []
for i in range(n):
    for j in range(i, n):
        num_to_coef.append((i, j))

coef_to_num = {coef: num for (num, coef) in enumerate(num_to_coef)}

right_part = np.concatenate(([y_mean], xy_mean, [xxy_mean[num_to_coef[i][0]][num_to_coef[i][1]]
                                                 for i in range(int(n * (n + 1) / 2))]))


main_matrix = np.concatenate((
    [np.concatenate(([1],
                     x_mean,
                     [xx_mean[num_to_coef[i][0]][num_to_coef[i][1]] for i in range(int(n * (n + 1) / 2))]))],
    [np.concatenate(([x_mean[i_s]],
                    xx_mean[i_s],
                    [xxx_mean[i_s][num_to_coef[i][0]][num_to_coef[i][1]] for i in range(int(n * (n + 1) / 2))]))
     for i_s in range(n)],
    [np.concatenate(([xx_mean[num_to_coef[num][0]][num_to_coef[num][1]]],
                    xxx_mean[num_to_coef[num][0]][num_to_coef[num][1]],
                    [xxxx_mean[num_to_coef[num][0]][num_to_coef[num][1]][num_to_coef[i][0]][num_to_coef[i][1]]
                     for i in range(int(n * (n + 1) / 2))]))
     for num in range(int(n * (n + 1) / 2))]
))

# print(np.linalg.det(main_matrix))
all_coefficients = np.dot(np.linalg.inv(main_matrix), right_part)
a_0 = all_coefficients[0]
a_i = all_coefficients[1:n+1]
a_ij = np.array([[
    0 if i>j else
    all_coefficients[1+n+coef_to_num[(i, j)]]
    for j in range(n)] for i in range(n)])

real_y = a_0 + np.array([np.sum([a_i[i]*xt[i][l] for i in range(n)]) for l in range(N)]) + \
         np.array([np.sum([[a_ij[i][j]*xt[i][l]*xt[j][l] for j in range(n)] for i in range(n)]) for l in range(N)])

R = 1 - np.sum((real_y-yes)**2)/np.sum((y_mean-yes)**2)
print(R)