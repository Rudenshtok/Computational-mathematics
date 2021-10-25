# Руденко Варвара 906 группа. Практическая задача вар 4

import numpy as np
import numpy.linalg as la

# Создаю матрицу
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

# распечатка собственных значений матрицы найденных с помощью np.linalg.eig
sob_znach = np.linalg.eig(A)[0]
#print(sob_znach)
# это нужно для проверки степенного метода

# распечатка min и max найденного с помощью np.linalg.eig
print("Результаты")
print("min_linal_eig = ", min(sob_znach))
print("max_linal_eig = ", max(sob_znach)) 

# создание рандомного вектора лдля начала итерации степенного метода
y0 = np.random.randint(-1, 1, 100)
# print(y0)

def power_method(A, x): # функция степенного метода
    lamb = 0
    epcilon = 10e-7 # точность, которая определяет, когда стоит останавливать метод
    while True:
        x_1 = A.dot(x)
        x_norm = la.norm(x_1)
        x_1 = x_1 / x_norm
        if (abs(lamb - x_norm) <= epcilon):
            break
        else:
            lamb = x_norm
            x = x_1
    return lamb

# полученное максимальное собственное значение степенным методом
my_lambda_max = power_method(A, y0)
print("my_lambda_max = ", my_lambda_max)

# Совпадает с тем, которое мы получили с помощью np.linalg.eig

# вычисляем минимальное собственное значение матрицы
B = A - my_lambda_max * np.eye(100)

my_lambda_2 = power_method(B,y0)
my_lambda_min = my_lambda_max - my_lambda_2
print("my_lambda_min = ", my_lambda_max - my_lambda_2)
# минимальное собственное значение совпадает с тем, что получено с помощью np.linalg.eig

# функция евклидовой нормы для матрицы
def Ecludic_norm(A):
    sum = 0
    for i in range(100):
        for j in range(100):
            sum += A[i][j]**2
    return sum**0.5

# написанная функция совпадает с встроенной np.norm
#print(la.norm(A), Ecludic_norm(A))

# находим число обусловленности по стандартной формуле
A_inv = la.inv(A)
mu = la.norm(A) * la.norm(A_inv)
print("число обусловленностей = ", mu)

# для самосопряжённой матрицы верно, что число обусловленности - отношение максимального собственного
# значения к минимальному. Моя матрица не самосопряжённая, поэтому такая формула не работает

# покажем это
#print(A.T) - несамосопряжённая матрица
#print(my_lambda_max/my_lambda_min) # получаем неверное число обусловлености 13.52 вместо 238.05