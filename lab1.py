import random
import math
def nu(x, theta):
    return theta[0] + theta[1]/(0.05 + x[0]**1) + theta[2]*x[0]*x[1] + theta[3]*x[1]**2 

def y(nu, e):
    return nu + e




theta = [1.1, 1.2, 20, 1.2]
n = 4 * len(theta)
ro = random.randint(5,15)/100
x = list()
for i in range(-2, 3, 1):
    for j in range(-2, 3, 1):
        x.append([i/2, j/2])

u = list()
u_ = list()
s_value = 0
for i in range(n):
    s = y(nu(x[i],theta), 0)
    u.append(s)
    s_value += s 
for i in range(n):
    u_.append(s_value/n)
w2 = 0
for i in range(len(u)):
    w2 += (u[i] - u_[i])**2;
w2 = w2/(n-1)
sigma = ro * w2
e = list()
for i in range(n):
    e.append(random.normalvariate(0, math.sqrt(sigma)))

y_list=list()
for i in range(n):
    y_list.append(y(nu(x[i], theta), e[i]))

for i in range(n):
    print(x[i], " ", y_list[i])
