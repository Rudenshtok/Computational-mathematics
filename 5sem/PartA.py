# Руденко Варвара 906 группа Практическая задача вар 4

import numpy as np
import numpy.linalg as la

#создаю матрицу, которую можно решить вручную
""" myA = [
       [1.0, -2.0, 3.0, -4.0],
       [3.0, 3.0, -5.0, -1.0],
       [3.0, 0.0, 3.0, -10.0],
       [-2.0, 1.0, 2.0, -3.0],
]
myB = [
       2.0,
       -3.0,
       8.0,
       5.0]

"""

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

# --- вывод системы на экран
def FancyPrint(A, B, selected):
    for row in range(len(B)):
        print("(", end='')
        for col in range(len(A[row])):
            print("\t{1:10.2f}{0}".format(" " if (selected is None or selected != (row, col)) else "*", A[row][col]),
                  end='')
        print("\t) * (\tX{0}) = (\t{1:10.2f})".format(row + 1, B[row]))


# --- end of вывод системы на экран


# --- перемена местами двух строк системы
def SwapRows(A, B, row1, row2):
    A[row1], A[row2] = A[row2], A[row1]
    B[row1], B[row2] = B[row2], B[row1]


# --- end of перемена местами двух строк системы


# --- деление строки системы на число
def DivideRow(A, B, row, divider):
    A[row] = [(a / divider) for a in A[row]]
    # for i in range(len(A[row])):
    #     A[row][i] = round(A[row][i], 3)
    B[row] /= divider
    #B[row] = round(B[row], 5)


# --- end of деление строки системы на число


# --- сложение строки системы с другой строкой, умноженной на число
def CombineRows(A, B, row, source_row, weight):
    A[row] = [(a + k * weight) for a, k in zip(A[row], A[source_row])]
    B[row] += B[source_row] * weight


# --- end of сложение строки системы с другой строкой, умноженной на число


# --- решение системы методом Гаусса (приведением к треугольному виду)
def Gauss(A, B):
    column = 0
    while (column < len(B)):

        #print("Ищем максимальный по модулю элемент в {0}-м столбце:".format(column + 1))
        current_row = None
        for r in range(column, len(A)):
            if current_row is None or abs(A[r][column]) > abs(A[current_row][column]):
                current_row = r
        if current_row is None:
            print("решений нет")
            return None
        #FancyPrint(A, B, (current_row, column))

        if current_row != column:
            #print("Переставляем строку с найденным элементом повыше:")
            SwapRows(A, B, current_row, column)
            #FancyPrint(A, B, (column, column))

        #print("Нормализуем строку с найденным элементом:")
        DivideRow(A, B, column, A[column][column])
        #FancyPrint(A, B, (column, column))

        #print("Обрабатываем нижележащие строки:")
        for r in range(column + 1, len(A)):
            CombineRows(A, B, r, column, -A[r][column])
        #FancyPrint(A, B, (column, column))

        column += 1

    #print("Матрица приведена к треугольному виду, считаем решение")
    X = [0 for b in B]
    for i in range(len(B) - 1, -1, -1):
        X[i] = B[i] - sum(x * a for x, a in zip(X[(i + 1):], A[i][(i + 1):]))

    #print("Получили ответ:")
    #print("\n".join("X{0} =\t{1:10.2f}".format(i + 1, x) for i, x in enumerate(X)))

    return X


# --- end of решение системы методом Гаусса (приведением к треугольному виду)



A_source = A.copy() # копируем матрицы, так как наш алгоритм изменяет их значения
B_source = B.copy()

#Решаю нашу систему, написанным алгоритмом
X = Gauss(A, B)
# печатаю полученное решение
print("решение - Гаусс:")
print("\n".join("X{0} =\t{1:10.2f}".format(i + 1, x) for i, x in enumerate(X)))

# для проверки решаем эту систему с помощью встроенного модуля np.linalg.solve
X_linalg = np.linalg.solve(A, B)
#print("\n".join("X{0} =\t{1:10.2f}".format(i + 1, x) for i, x in enumerate(X_linalg)))

#print(X == X_linalg) проверила, что результаты совпадают, следовательно, получительное решение верно

A_source = np.array(A_source, dtype=np.float)

# нахожу невязку (ошибку)
r = A_source.dot(X) - B_source
print("Невязка по каждому элементу матрицы")
print(r)

# нахожу норму невязки
mu = 238.05225200738846
mu_new = la.norm(r)
print("норма невязки = ",mu_new, "   ", "число обусловленностей = ", mu)

print("норма решения = ", la.norm(X))
print("отношение нормы невязки к норме решения = ", la.norm(r)/la.norm(X))