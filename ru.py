import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.misc import derivative
from numpy.linalg import linalg


def E(i, n):
    h = 1/n
    return lambda x: max(0, (1-abs((x-i*h)/(h))))

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
    MB = np.zeros((n, n))
    for i in range (0, n):
        for j in range(0, n):
            MB[i][j] = BMatrixElem(i, j, n, fa, fb, fc, beta)

    return MB


def L(ff, j, n, fa, fb, fc, beta, gamma, n1):
    v = E(j, n)
    I = quad((lambda x: ff(x) * v(x)), 0, 1)[0]
    return I - gamma*v(0) - BShift(j, n, fa, fb, fc, (lambda x: n1*E(n,n)(x)), beta)

def LMatrix(n, f, fa, fb, fc, beta, gamma, n1):
    ML = np.zeros(n)
    for j in range(0, n):
        ML[j] = L(f, j, n, fa, fb, fc, beta, gamma, n1)
    return ML

def solveGalerkin(fa, fb, fc, ff, beta, gamma, n1, n):
    MB = BMatrix(n, fa, fb, fc, beta)
    ML = LMatrix(n, ff, fa, fb, fc, beta, gamma, n1)
    print(MB)
    print(ML)
    lp = np.linspace(0, 1, 100)

    res = linalg.solve(MB, ML)
    print(res)

    


a = lambda x: 1
b = lambda x: 0
c = lambda x: 0
f = lambda x: 0
Beta = 0
Gamma = 0
N1 = 1.0

solveGalerkin(a, b, c, f, Beta, Gamma, N1, 5)
# print (list(map(E(2, 10), lp)))

# es = [(lambda x: E(i, 5)(x) * res[i]) for i in range(0, 5+1)]


# plt.plot(lp, np.fromiter(map(E(2, 10), lp), dtype=np.float_))
# plt.plot(lp, np.fromiter(map(E(5, 5), lp), dtype=np.float_))

# plt.show()
