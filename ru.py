import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.misc import derivative
from numpy.linalg import linalg
from functools import reduce


def E(i, n):
    h = LEN/n
    return lambda x: max(0, (1-abs((-A + x-i*h)/(h))))

def dE(i, n):
    return lambda x: derivative(E(i, n), x, dx = 1e-6)

def BMatrixElem(i, j, n, fa, fb, fc, beta):
    v = E(j, n)
    u = E(i, n)
    dv = dE(j, n)
    du = dE(i, n)
    I1 = quad((lambda x: fa(x) * du(x) * dv(x)), A, B, limit=QUAD_LIMIT, points=QUAD_POINTS)[0]
    I2 = quad((lambda x: fb(x) * du(x) * v(x)), A, B, limit=QUAD_LIMIT, points=QUAD_POINTS)[0]
    I3 = quad((lambda x: fc(x) * u(x) * v(x)), A, B, limit=QUAD_LIMIT, points=QUAD_POINTS)[0]
    return (-v(A)*u(A)*beta - I1 + I2 + I3)

def BShift(j, n, fa, fb, fc, shift, beta):
    v = E(j, n)
    dv = dE(j, n)
    u = shift
    du = lambda x: derivative(u, x, dx=1e-6)
    I1 = quad((lambda x: fa(x) * du(x) * dv(x)), A, B, limit=QUAD_LIMIT, points=QUAD_POINTS)[0]
    I2 = quad((lambda x: fb(x) * du(x) * v(x)), A, B, limit=QUAD_LIMIT, points=QUAD_POINTS)[0]
    I3 = quad((lambda x: fc(x) * u(x) * v(x)), A, B, limit=QUAD_LIMIT, points=QUAD_POINTS)[0]
    return (-v(A)*u(A)*beta - I1 + I2 + I3)


def BMatrix(n, fa, fb, fc, beta):
    MB = np.zeros((n, n))
    for i in range (0, n):
        for j in range(0, n):
            MB[i][j] = BMatrixElem(i, j, n, fa, fb, fc, beta)

    return MB


def L(ff, j, n, fa, fb, fc, beta, gamma, n1):
    v = E(j, n)
    I = quad((lambda x: ff(x) * v(x)), A, B)[0]
    return I - gamma*v(A) - BShift(j, n, fa, fb, fc, (lambda x: n1*E(n,n)(x)), beta)

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


    res = linalg.solve(MB, ML)

    res = np.append(res, n1)

    # print(res)

    return res





def FinalFunc(n, res, x):
    val = 0
    for i in range(0, n+1):
        val += res[i] * E(i, n)(x)
    return val

A = -3*np.pi
B = 4*np.pi
LEN = B-A


a = lambda x: 1
b = lambda x: 0
c = lambda x: 1
f = lambda x: 0
Beta = 0
Gamma = 0
N1 = np.cos(B)
N = int(40)

PLOT_RANGE = [A-0.05*LEN, B+0.05*LEN, -1.5, 5]
QUAD_LIMIT = 270
QUAD_POINTS = np.linspace(A, B, N)



print("Calculating cos")
cosinus = solveGalerkin(a, b, c, f, Beta, Gamma, N1, N)

# N1 = np.sin(B)


# print("Calculating sin")
# sinus = solveGalerkin(a, b, c, f, Beta, Gamma, N1, N)

a = lambda x: 1
b = lambda x: 0
c = lambda x: -1
f = lambda x: 0
Beta = 0
Gamma = 0
N1 = np.exp(B)
N = int(40)


print("Calculating exp")
exponent = solveGalerkin(a, b, c, f, Beta, Gamma, N1, N)

lp = np.linspace(A, B, 100)
# plt.plot(lp, np.fromiter(map((lambda x: FinalFunc(N, sinus, x)), lp), dtype=np.float_))
plt.plot(lp, np.fromiter(map((lambda x: FinalFunc(N, cosinus, x)), lp), dtype=np.float_))
plt.plot(lp, np.fromiter(map((lambda x: FinalFunc(N, exponent, x)), lp), dtype=np.float_))
plt.axis(PLOT_RANGE)
plt.show()