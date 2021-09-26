import random
import math
import numpy
import matplotlib.pyplot as plt

def f1(x):
    return 1

def f2(x):
    return 1/(0.05 + x[0]**1)

def f3(x):
    return x[1]**2

def f4(x):
    return x[0]*x[1]

def etta(x, theta):
    return theta[0]*f1(x) + theta[1]*f2(x) + theta[2]*f3(x) + theta[3]*f4(x)

def y(etta, e):
    return etta + e

def generation():
    # Генерируем сетку
    for i in range(-2, 2, 1):
        for j in range(-2, 2, 1):
            x.append([i/2, j/2])

    u = list()
    u_ = list()

    # Считаем незашумленный отклик
    s_value = 0
    for i in range(n):
        s = y(etta(x[i], theta), 0)
        u.append(s)
        s_value += s 

    # Считаем вектор среднего значения сигнала
    for i in range(n):
        u_.append(s_value / n)

    # Считаем мощность сигнала
    w2 = 0
    for i in range(len(u)):
        w2 += (u[i] - u_[i])**2;
    w2 = w2 / (n - 1)

    # Дисперсия помехи е
    sigma = ro * w2

    # Моделирование помехи 
    e = list()
    for i in range(n):
        e.append(random.normalvariate(0, math.sqrt(sigma)))

    # Считаем зашумленный отклик
    for i in range(n):
        y_list.append(y(etta(x[i], theta), e[i]))

    # Вывод полученных результатов
    for i in range(n):
        print(x[i], " ", y_list[i])
    
    return sigma


def create_plot():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    X = tuple([xi[0] for xi in x])
    Y = tuple([xi[1] for xi in x])
    Z = list()
    for i in range(n):
        Z.append(y(etta(x[i], theta), 0))
    Z = tuple(Z)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Theta')
    ax.plot_trisurf(X, Y, Z)
    ax.set_title("График зависимости незашумленного отклика от факторов")

    plt.show()


def evaluate_theta(X):
    # Строим матрицу X
    for i in range(n):
        X.append([f1(x[i]), f2(x[i]), f3(x[i]), f4(x[i])])

    X = numpy.array(X)
    XT = X.transpose()

    XtX = numpy.matmul(XT,X)

    inv_XtX = numpy.linalg.inv(XtX)

    S2 = numpy.matmul(inv_XtX, XT)
    res = numpy.matmul(S2, y_list)

    print("^theta = ", res)
    return res


def check_adequate(X, res, sigma):
    E = numpy.subtract(y_list, numpy.matmul(X, res))
    ET = E.transpose()
    sigma2 = numpy.matmul(ET,E) / 12
    
    print(f"F = {sigma2/sigma} <= F_T = 2.2962")
    print((sigma2/sigma) <= 2.296)


if __name__=="__main__":
    # Lab 1
    theta = [1.1, 1.2, 1.2, 20]
    n = 4 * len(theta)
    ro = random.randint(5,15) / 100
    x = list()
    y_list = list()

    sigma = generation()

    # Lab 2
    X = list()
    res = evaluate_theta(X)
    check_adequate(X, res, sigma)

    # Plot 
    create_plot()
   