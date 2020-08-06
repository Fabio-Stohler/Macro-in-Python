import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)

from EGM import exogTrans, SolveEGM, N, interp, G, Prices, Lbar, alpha, beta, delta, gamma, rho
from EGM import grid as A
from EGM import gridsize as Asize
from Support import GetStationaryDist

from scipy.optimize import brentq
import matplotlib.pyplot as plt

def MakeTransMat(G):
    # Rows of T correspond to where we are going, cols correspond to where we are coming from
    T = np.zeros((N*Asize,N*Asize))
    for j in range(N):
        x, i = interp(G[j],A,A)
        p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])
        p = np.minimum(np.maximum(p,0.0),1.0)
        sj = j*Asize
        for k in range(N):
            sk = k * Asize
            T[sk + i,sj+np.arange(Asize)]= p * exogTrans[k,j]
            T[sk + i - 1,sj+np.arange(Asize)] = (1.0-p)* exogTrans[k,j]

    assert np.allclose(T.sum(axis=0), np.ones(N*Asize))
    return T



def Agg_K(D):
    return (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()

History = []

def Check_K(K):
    Pr = Prices(K)
    G[:] = SolveEGM(G,Pr)
    K_implied = Agg_K(GetStationaryDist(MakeTransMat(G)))
    History.append([K,K_implied])
    return K_implied - K


KCompleteMarkets =  ((1/beta-1+delta)/alpha)**(1./(alpha-1)) * Lbar
KLarge = 48.0
Kstar = brentq(Check_K,KCompleteMarkets,KLarge,xtol=1e-16)

Pr = Prices(Kstar)
G[:] = SolveEGM(G,Pr)
D = GetStationaryDist(MakeTransMat(G))
# Kstar = Agg_K(D)





# This material below runs only if this file is executed as a script
if __name__ == "__main__":
    normfactor = np.hstack((1.0,np.diff(A)))[np.newaxis,:]
    plt.figure()
    Density = D.reshape(N,Asize)/normfactor
    plt.plot(A,Density.T)
    plt.xlabel('Current Assets')
    plt.ylabel('Density')
    plt.legend(['Low endow', 'High endow'])
    plt.show()

    History = np.array(History)
    iH = np.argsort(History[:,0])
    KSupply = History[:,1][iH]
    KDemand = History[:,0][iH]
    R = [Prices(K).R for K in KDemand]
    plt.figure()
    lkd, = plt.plot(KDemand,R,label = 'Capital demand')
    lks, = plt.plot(KSupply,R, label = 'Capital supply')
    plt.xlabel('Capital stock')
    plt.ylabel('R')
    plt.legend(handles = [lkd,lks])
    plt.xlim([40.,60.])
    plt.show()
