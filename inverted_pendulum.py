import numpy as np
from math import sqrt, sin, cos
import matplotlib.pyplot as plt
from control import lqr
import slycot
import imageio

Mc = 2.4
mp = .23
lp = 1.
g = 9.8

k = .1
c = .2
m = 1.5

def LQR():
    A = np.array([[0,    1],
                  [g/lp, 0]])

    B = np.array([[0],
                  [1]])
    
    Q = np.array([[3000., 0],
                  [0,   1.]])
    R = .1

    K, S, E = lqr(A, B, Q, R)

    return np.array(K)


def func(states, t, time_count):
    theta, theta_dot = states
    K = LQR()
    k1 = K[0][0]
    k2 = K[0][1]

    reset = time_count % 31

    ref = .5 if reset < 15 else 1.5
    N = 174.5

    theta_ddot = (g/lp) * sin(theta) + ((N*ref) - (k1*theta + k2*theta_dot))
    return np.array([theta_dot, theta_ddot])


def rk4(f, u0, t0, tf, n):
    t = np.linspace(t0, tf, n+1)
    u = np.array((n+1)*[u0])
    h = t[1]-t[0]
    t_count = 0

    for i in range(n):
        k1 = h * f(u[i], t[i], t_count)
        k2 = h * f(u[i] + 0.5 * k1, t[i] + 0.5*h, t_count)
        k3 = h * f(u[i] + 0.5 * k2, t[i] + 0.5*h, t_count)
        k4 = h * f(u[i] + k3, t[i] + 0.5*h, t_count)
        u[i+1] = u[i] + (k1 + 2*(k2 + k3) + k4) / 6
        t_count += 1
    return u, t


u, t = rk4(func, np.array([-.5, 0.]), 0., 5., 100)
x1, v1 = u.T
print(x1[-1], v1[-1])
plt.plot(t, x1*57.296)
plt.grid('on')
plt.show()


ct = 1
images = []

for i in x1[:]:
    x0 = 0
    y0 = 0

    x1 = lp*sin(i)
    y1 = lp*cos(i)

    plt.figure()
    plt.plot([-.2, .2], [0, 0], 'black', linewidth=5)
    plt.plot([x0, x1], [y0, y1], 'red', linewidth=3)
    plt.plot(x1, y1, 'o', markersize=15)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.grid()
    filename = '/home/tyler/Pictures/' + str(ct) + '.png'
    plt.savefig(filename)
    images.append(imageio.imread(filename))
    plt.close()
    ct += 1

imageio.mimsave('/home/tyler/Videos/Inv_Pendulum.gif', images)
