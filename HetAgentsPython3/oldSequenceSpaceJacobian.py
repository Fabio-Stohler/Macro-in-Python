import autograd.numpy as np
np.set_printoptions(suppress=True,precision=4)

from Aiyagari import MakeTransMat, Kstar, alpha, beta, delta, gamma, rho, Agg_K, G, D
from EGM import grid as A
from EGM import gridsize as Asize
from EGM import eulerBack, get_c, Prices, N, interp, exogTrans, SolveEGM
from MITshocks import UpdateAggs

import matplotlib.pyplot as plt

########### Now we already have the aiyagari solution need to construct MIT shock economy
### Pick horizon for impulse response function with TimePeriods

TimePeriods = 200
Z_ss = 1.0

D_SS = D.copy()
G_SS = G.copy()


def BackwardIteration(Kguess,Zpath):
    calD = np.zeros((N*Asize,TimePeriods))
    calY = np.zeros(TimePeriods)

    G = G_SS
    for u in range(TimePeriods):
        K = Kguess[TimePeriods-u-1]
        Z = Zpath[TimePeriods-u-1]

        if u == 0:
            Z_P = Z_ss
        else:
            Z_P = Zpath[TimePeriods-u]

        if u == TimePeriods-1:
            K_m = Kstar
        else:
            K_m = Kguess[TimePeriods-u-2]

        Pr_P = Prices(K,Z_P)
        Pr = Prices(K_m,Z)
        G = eulerBack(G,Pr,Pr_P)[0]

        calD[ :,  u] = (MakeTransMat(G)@D_SS - D_SS)/eps
        calY[ u ] = 0.0  # the individual outcomes do not depend on the policy rules in this case

    return calD,calY

def ForwardIteration():
    calP = np.zeros((N*Asize,TimePeriods-1))
    calP[:,0] = np.tile(A,(1,N))
    Lambda_SS = MakeTransMat(G_SS).transpose()
    for i in range(1,TimePeriods-1):
        calP[:,[i]] = Lambda_SS @ calP[:,[i-1]]

    return calP


K = Kstar*np.ones(TimePeriods)
Z = Z_ss*np.ones(TimePeriods)
eps = 1e-4

Kpert = K.copy()
Kpert[-1] += eps

Zpert = Z.copy()
Zpert[-1] += eps

calD_K,calY_K = BackwardIteration(Kpert,Z)
calD_Z,calY_Z = BackwardIteration(K,Zpert)

calP = ForwardIteration()



# --- Make the Fake News Matrices ---
calF_K = np.zeros((TimePeriods,TimePeriods))
calF_K[0,:] = calY_K
calF_K[1:,:] = calP.transpose() @ calD_K

calF_Z = np.zeros((TimePeriods,TimePeriods))
calF_Z[0,:] = calY_Z
calF_Z[1:,:] = calP.transpose() @ calD_Z

# --- Fake news recursion ---
Jac_K = calF_K.copy()
for t in range(1,TimePeriods):
    Jac_K[t,1:] += Jac_K[t-1,:-1]

Jac_Z = calF_Z.copy()
for t in range(1,TimePeriods):
    Jac_Z[t,1:] += Jac_Z[t-1,:-1]


#-- testing

def F(K,Z):
    return UpdateAggs(G_SS,D_SS,K,Z)[2]


Kpert = K.copy()
Zpert = Z.copy()

NumJacCol = 0
Kpert[NumJacCol] += eps

NumJac_K = (F(Kpert,Z)-F(K,Z))/eps
plt.plot(np.vstack((Jac_K[:,NumJacCol],NumJac_K)).T)
plt.show()


# Notes:
# NumJacCol = 0 leads to close to the right answer with wrong sign.
# NumJacCol = 3 looks quite wrong.
# This might indicate a problem with the backward iteration step
# I have not understood or implemented footnote 21
# I wonder if I have some details of the timing wrong in the backward iteration
