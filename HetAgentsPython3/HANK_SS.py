import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)

from scipy.optimize import brentq
from Support import GetStationaryDist

from HANK_EGM import SolveEGM, Pr0, Mbar, ubar, G, A, Asize, N, B, Prices, interp, delta



def MakeTransMat(G, M):
    return MakeTransMat_Savings(G) @ MakeTransMat_Emp(M)

def MakeTransMat_Savings(G):
    # Rows of T correspond to where we are going, cols correspond to where we are coming from
    T = np.zeros((N*Asize,N*Asize))
    for j in range(N):
        x, i = interp(G[j],A,A)
        p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
        p = np.minimum(np.maximum(p,0.0),1.0)
        sj = j*Asize
        T[sj + i,sj+np.arange(Asize)]= p
        T[sj + i - 1,sj+np.arange(Asize)] = (1.0-p)

    assert np.allclose(T.sum(axis=0), np.ones(N*Asize))
    return T

def MakeTransMat_Emp(M):
    return np.kron(np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]]), np.eye(Asize))

def Agg_Assets(D):
    return (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()



History = []

def Check_Assets(R):
    Pr = Prices(R)
    G[:] = SolveEGM(G,Pr)
    Assets_implied = Agg_Assets(GetStationaryDist(MakeTransMat(G,Pr.M)))
    History.append([R,Assets_implied])
    print("Checking R = {0}, assets implied = {1}".format(R,Assets_implied))
    return Assets_implied - B


Rstar = brentq(Check_Assets,1.000,1.0175,xtol = 1e-14)
Pr = Prices(Rstar)
G[:] = SolveEGM(G,Pr)
D = GetStationaryDist(MakeTransMat(G,Pr.M))
