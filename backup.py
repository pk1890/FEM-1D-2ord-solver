import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.misc import derivative
from numpy.linalg import linalg



def E(i, n):
    h = 1/n
    return lambda x: (1-abs((x-i*h)/(h)) if (x > (i-1)*h and x < (i+1)*h) else 0)

def dE(i, n):
    return lambda x: derivative(E(i, n), x, dx = 1e-6)

def BMatrixElem(i, j, n, fa, fb, fc, beta):
    v = E(j, n)
    u = E(i, n)
    dv = dE(j, n)
    du = dE(i, n)
    I1 = quad((lambda x: fa(x) * du(x) * dv(x)), 0, 1)[0]
    I2 = quad((lambda x: fb(x) * du(x) * v(x)), 0, 1)[0]
    I3 = quad((lambda x: fc(x) * u(x) * v(x)), 0, 1)[0]
    return (-v(0)*u(0)*beta - I1 + I2 + I3)

def BShift(j, n, fa, fb, fc, shift, beta):
    v = E(j, n)
    dv = dE(j, n)
    u = shift
    du = lambda x: derivative(u, x, dx=1e-6)
    I1 = quad((lambda x: fa(x) * du(x) * dv(x)), 0, 1)[0]
    I2 = quad((lambda x: fb(x) * du(x) * v(x)), 0, 1)[0]
    I3 = quad((lambda x: fc(x) * u(x) * v(x)), 0, 1)[0]
    return (-v(0)*u(0)*beta - I1 + I2 + I3)


def BMatrix(n, fa, fb, fc, beta):
    MB = np.zeros((n+1, n+1))
    for i in range (0, n+1):
        for j in range(0, n+1):
            MB[i][j] = BMatrixElem(i, j, n, fa, fb, fc, beta)

    return MB


def L(f, j, n, fa, fb, fc, beta, gamma, n1):
    v = E(j, n)
    I = quad((lambda x: f(x) * v(x)), 0, 1)[0]
    return I - gamma*v(0) - BShift(j, n, fa, fb, fc, (lambda x: n1*E(n,n)(x)), beta)

def LMatrix(n, f, fa, fb, fc, beta, gamma, n1):
    ML = np.zeros(n+1)
    for j in range(0, n+1):
        ML[j] = L(f, j, n, fa, fb, fc, beta, gamma, n1)
    return ML

def solveGalerkin(n, fa, fb, fc, ff, beta, gamma, n1):
    MB = BMatrix(n, fa, fb, fc, beta)
    ML = LMatrix(n, ff, fa, fb, fc, beta, gamma, n1)
    lp = np.linspace(0, 1, n)

    print(linalg.solve(MB, ML))

    # print (list(map(E(2, 10), lp)))

    plt.plot(lp, np.fromiter(map(E(2, 10), lp), dtype=np.float_))
    plt.plot(lp, np.fromiter(map(dE(2, 10), lp), dtype=np.float_))

    plt.show()

a = lambda x: 1
b = lambda x: 0
c = lambda x: 0
f = lambda x: 0
Beta = 0.0
Gamma = 0.0
N1 = 1.0

solveGalerkin(100, a, b, c, f, Beta, Gamma, N1)