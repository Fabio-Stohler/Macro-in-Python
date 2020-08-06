import autograd.numpy as np
np.set_printoptions(suppress=True,precision=4)

from Aiyagari import MakeTransMat, Kstar, alpha, beta, delta, gamma, rho, Agg_K, G, D
from EGM import grid as A
from EGM import gridsize as Asize
from EGM import eulerBack, get_c, Prices, N, interp, exogTrans, SolveEGM

import matplotlib.pyplot as plt

########### Now we already have the aiyagari solution need to construct MIT shock economy
### Pick horizon for impulse response function with TimePeriods

TimePeriods = 200
OffeqPathTime = 4
Z_ss = 1.0
Kguess = Kstar*np.ones(TimePeriods)
Zpath = np.hstack((1.01*np.ones(OffeqPathTime),Z_ss*np.ones(TimePeriods-OffeqPathTime)))



def UpdateAggs(initialpol,initialdis,Kguess,Zpath):

    TimePeriods = len(Kguess)
    apols = np.zeros((TimePeriods,N*Asize))
    devol = np.zeros((TimePeriods,N*Asize))
    aggK  = np.zeros(TimePeriods)

    G = initialpol
    for i in range(TimePeriods-1,-1,-1):
        K,Z_P = Kguess[i],Zpath[i]
        K_m,Z = Kguess[i-1],Zpath[i-1]
        Pr_P = Prices(K,Z_P)
        Pr = Prices(K_m,Z)
        G = eulerBack(G,Pr,Pr_P)[0]
        apols[i-1] = G.reshape(-1)


    D = initialdis
    aggK[0] = Agg_K(initialdis)
    devol[0] = D
    for i in range(0,TimePeriods-1):
        G = apols[i]
        trans = MakeTransMat(G.reshape((N,Asize)))
        D = trans@D
        aggK[i+1] = Agg_K(D)
        devol[i+1] = D


    return apols,devol,aggK

def equilibrium(Kguess,Zpath,initialpol,initialdis):
    G_ss,D_ss = initialpol,initialdis
    for i in range(1,100):
        apols,devol,aggK = UpdateAggs(G_ss,D_ss,Kguess,Zpath)
        dif = max(abs(Kguess-aggK))
        print(dif)
        if dif < 1e-6:
            return apols,devol,aggK
        Kguess = 0.2*aggK + 0.8*Kguess

    print("Did not converge")


if __name__ == "__main__":
    apols,devol,aggK = equilibrium(Kguess,Zpath,G,D)


    plt.figure()
    plt.plot(range(TimePeriods-1),aggK[1:]/aggK[0]-1)
    plt.xlabel('Quarter')
    plt.ylabel('% deviation of K from SS')
    plt.show()
