# Руденко Варвара 906 группа Практическая задача вар 4

import numpy as np
import numpy.linalg as la

# создание матрицы моего варианта
first = [1 for i in range(1,101)]
second = [1,10,1]
for i in range(97):
    second.append(0)


AAA = [first, second]

for i in range(1,98):
    massiv = [0 for j in range(i)]
    massiv.append(1)
    massiv.append(10)
    massiv.append(1)
    while len(massiv) != 100:
        massiv.append(0)
    AAA.append(massiv)


last = [0 for i in range(98)]
last.append(1)
last.append(1)

AAA.append(last)
A = np.array(AAA)


# распечатка самой матрицы

# for i in range(100):
#     print(A[i])

# создаём столбец свободных коэффициентов
B = [i for i in range(1,101)]

# для проверки решаем эту систему с помощью встроенного модуля np.linalg.solve
X_linalg = np.linalg.solve(A, B)
#print("\n".join("X{0} =\t{1:10.2f}".format(i + 1, x) for i, x in enumerate(X_linalg)))

def Zeidel(A, B, precision): # функция, которая решает систему алгоритмом Зейделя
    LD = np.zeros_like(A) # создаём матрицу L + D
    U = np.zeros_like(A)  # создаём матрицу U
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i >= j:
                LD[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]

    B_iter = -la.inv(LD).dot(U) # получили матрицу -(L + D)^-1 * U
    F = la.inv(LD).dot(B) # получили (L + D)^-1 * B
    x0 = np.zeros(100) # задаём начальное решение нулём
    iter = 1 # счётчик кол-ва итераций
    while True:
        x_new = B_iter.dot(x0) + F # по формуле Зейделя считаем новое решение
        if (la.norm(x_new - x0) < precision): # пока норма предыдцщего и текущего x больше нашей точности, продолжаем
            break
        else:
            x0 = x_new
            iter += 1
    return x0, iter

A_source = A.copy() # копируем матрицы, так как наш алгоритм изменяет их значения
B_source = B.copy()

X_Zeidel, iter = Zeidel(A, B, 10e-4)
print("\n".join("X{0} =\t{1:1.2f}".format(i + 1, x) for i, x in enumerate(X_Zeidel)))
print("количество итераций = ", iter)

r = A_source.dot(X_Zeidel) - B_source
print("Невязка по каждому элементу матрицы")
print(r)

# находим число обусловленности по стандартной формуле
A_inv = la.inv(A_source)
mu = la.norm(A_source) * la.norm(A_inv)
print("число обусловленностей = ", mu)

# нахожу норму невязки
mu_new = la.norm(r)
print("норма невязки = ", mu_new)

print("норма решения = ", la.norm(X_Zeidel))
print("отношение нормы невязки к норме решения = ", la.norm(r)/la.norm(X_Zeidel))