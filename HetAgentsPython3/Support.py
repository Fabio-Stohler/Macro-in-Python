import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)

import warnings

def GetStationaryDist(T):
    eval,evec = np.linalg.eig(T)
    i = np.argmin(np.abs(eval-1.0))
    D = np.array(evec[:,i]).flatten()
    assert np.max(np.abs(np.imag(D))) < 1e-6
    D = np.real(D)  # just recasts as float
    return D/D.sum()




# Solve the system
def SolveSystem(A,B,C,E,P0=None):
    # Solve the system using linear time iteration as in Rendahl (2017)
    print("Solving the system")
    MAXIT = 1000
    if P0 is None:
        P = np.zeros(A.shape)
    else:
        P = P0

    S = np.zeros(A.shape)

    for it in range(MAXIT):
        P = -np.linalg.lstsq(B+A@P,C,rcond=None)[0]
        S = -np.linalg.lstsq(B+C@S,A,rcond=None)[0]
        test = np.max(np.abs(C+B@P+A@P@P))
        if it % 20 == 0:
            print(test)
        if test < 1e-7:
            break


    if it == MAXIT-1:
        warnings.warn('LTI did not converge.')


    # test Blanchard-Kahn conditions
    if np.max(np.linalg.eig(P)[0])  >1:
        raise RuntimeError("Model does not satisfy BK conditions -- non-existence")

    if np.max(np.linalg.eig(S)[0]) >1:
        raise RuntimeError("Model does not satisfy BK conditions -- mulitple stable solutions")

    # Impact matrix
    #  Solution is x_{t}=P*x_{t-1}+Q*eps_t
    Q = -np.linalg.inv(B+A@P) @ E

    return P, Q


# I use this in debugging to test the automatic derivatives
def FiniteDifferenceDerivative(func,x,i,eps = 1e-5):

    y = func(x)

    eps = 1e-5
    xx = x.copy()
    xx[i] += eps
    yup = func(xx)
    dup = (yup - y) / eps

    xx[i] -= 2*eps
    ydn = func(xx)
    ddn = -(ydn - y) / eps

    d = 0.5*( dup + ddn)

    d = np.where( np.abs(dup-ddn)/(np.abs(d)+eps) > eps, np.nan,d )

    return d
