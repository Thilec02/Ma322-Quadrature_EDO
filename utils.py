import numpy as np


def EulerExplicit(Yn, f, tau):
    return Yn + tau * f(Yn)


def ode_RK4(f, a, b, ic, N):
    h = (b - a) / N  # step size if h is constant
    Ly = np.empty((N, np.size(ic)), dtype=float)
    Ly[0, :] = ic
    for i in range(N - 1):
        # if h isn't constant, we use h=t[i+1]-t[i]
        k1 = h * f(Ly[i, :])
        y1 = Ly[i, :] + 1 / 2 * k1
        k2 = h * f( y1)
        y2 = Ly[i, :] + 1 / 2 * k2
        k3 = h * f(y2)
        y3 = Ly[i, :] + k3
        k4 = h * f(y3)
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        Ly[i + 1, :] = Ly[i, :] + k
    return Ly


def RKF4Butcher(f, Y, tau, M):
    beta = np.array([[0, 0, 0, 0, 0],
                     [1/4, 0, 0, 0, 0],
                     [3/32, 9/32, 0, 0, 0],
                     [1932/2197, -7200/2197,
                      7296/2197, 0, 0],
                     [439/216, -8, 3680/513, -845/4104, 0],
                     [-8/27, 2, -3544/2565, 1859/4104, -11/40]],
                     dtype=object)
    gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    K = []
    for i in range(M-1):
        if i == 1:
            K.append(f(Y))
        else:
            K.append(f(Y + tau*sum(beta[i][j]*K[j] for j in range(i))))

    Y = Y+tau*sum(gamma[j]*K[j] for j in range(M-1))

    return Y


def stepRK45(f, Y, tau_next, M, epsilon_max):
    beta = np.array([[0, 0, 0, 0, 0],
                     [1 / 4, 0, 0, 0, 0],
                     [3 / 32, 9 / 32, 0, 0, 0],
                     [1932 / 2197, -7200 / 2197,
                      7296 / 2197, 0, 0],
                     [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
                     [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40]],
                    dtype=object)
    gamma = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    gammab = np.array([25/216, 0, 1408/2565, 2197, 4104, -1/5, 0])

    K = []

    for i in range(M - 1):
        if i == 1:
            K.append(f(Y))
        else:
            K.append(f(Y + tau_next * sum(beta[i][j] * K[j] for j in range(i))))

    E = tau_next * sum((gamma[i] - gammab[i]) * K[i] for i in range(M - 1))
    epsilon = np.max(abs(E))
    while epsilon > epsilon_max:
        Y = RKF4Butcher(f, Y, tau_next, M)
        E = tau_next * sum((gamma[i] - gammab[i]) * K[i] for i in range(M - 1))
        epsilon = np.max(abs(E))
        e = 0.9 * (epsilon_max / epsilon) ** (1 / 5)
        if e<0.1:
            tau_next *=0.1
        if e>5:
            tau_next *=5
        else:
            tau_next *=e

    return Y, tau_next
