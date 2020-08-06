import autograd.numpy as np
from autograd import jacobian
from autograd.extend import defvjp, primitive
np.set_printoptions(suppress=True,precision=4)

from Aiyagari import MakeTransMat, Kstar, alpha, beta, delta, gamma, rho, Agg_K, G, D
from EGM import grid as A
from EGM import gridsize as Asize
from EGM import eulerBack, get_c, Prices, N, interp, exogTrans

from SolveQZ import solve
from RBC import IRF_RBC

import matplotlib.pyplot as plt

# Notes:
# In this module, we will treat G as a vector for the most part and then
# reshape it into the appropriate matrix as necessary.  This is done in order
# to more easily take the Jacobian w.r.t. G (i.e. Jacobian where input is a vector
# makes more sense than Jaboian where input is a matrix).





# rewrite eulerBack to deliver residual function
def eulerResidual(G,G_P,Pr,Pr_P):
# The arguments are the savings policy rules in the current and next period
# parameterized as the a values associated with savings grid for a'

    a, c = eulerBack(G_P.reshape(N,Asize),Pr,Pr_P)
    c2 = get_c(G.reshape(N,Asize),Pr,CurrentAssets = a)
    return (c/c2 - 1).reshape(-1)


# get residuals of distribution of wealth equations
@primitive # tell autograd not to differentiate this function (we will do it by hand)
def wealthResidual(G,D_L,D):
    return (D - MakeTransMat(G.reshape(N,Asize)) @ D_L)[1:]  # drop one redundant equation (prob distribution sums to 1)

def AggResidual(D,K,Z_L,Z,epsilon):
    return np.hstack((K - Agg_K(D), np.log(Z) - rho*np.log(Z_L)-epsilon))

def F(X_L,X,X_P,epsilon):
    # Bundle the equations of the model

    # Step 1: unpack
    m = N*Asize
    G_L,D_L,K_L,Z_L = X_L[:m], X_L[m:(2*m-1)], X_L[2*m-1], X_L[2*m]
    G  ,D  ,K  ,Z   = X[:m]  , X[m:(2*m-1)],   X[2*m-1],   X[2*m]
    G_P,D_P,K_P,Z_P = X_P[:m], X_P[m:(2*m-1)], X_P[2*m-1], X_P[2*m]

    D_L = np.hstack((1-D_L.sum(), D_L))
    D = np.hstack((1-D.sum(), D))
    D_P = np.hstack((1-D_P.sum(), D_P))

    # Step 2: prices
    Pr = Prices(K_L,Z)
    Pr_P = Prices(K,Z_P)

    # Step 3: bundle equations
    return np.hstack( (eulerResidual(G,G_P,Pr,Pr_P), wealthResidual(G,D_L,D), AggResidual(D,K,Z_L,Z,epsilon) ) )

# pack
X_SS = np.hstack((G.reshape(-1), D[1:], Kstar, 1.0))
epsilon_SS = 0.0

# test steady state
assert np.allclose(F(X_SS,X_SS,X_SS,epsilon_SS) , np.zeros(2*N*Asize+1))

# Before we linearize we need to define a derivative for wealthResidual

def Deriv_MakeTransMat(G):
    # Create a 3-D array TD
    # TD[i,j,k] is the derivative of the transition probability T[i,j] with respect to G[k] (where G has been flattened)
    G = G.reshape(N,Asize)
    TD = np.zeros((N*Asize,N*Asize,G.size))
    for j in range(N):
        x, i = interp(G[j],A,A)
        p = (A-G[j,i-1]) / (G[j,i] - G[j,i-1])

        dpdGLeft = (p-1) / (G[j,i] - G[j,i-1])
        dpdGRight = - p / (G[j,i] - G[j,i-1])

        dpdGLeft = np.where( (p > 0) & (p<1) , dpdGLeft, 0.0 )
        dpdGRight = np.where( (p > 0) & (p<1) , dpdGRight, 0.0 )


        sj = j*Asize
        for k in range(N):
            sk = k * Asize
            TD[sk + i,sj+np.arange(Asize), sj + i ] += dpdGRight * exogTrans[k,j]
            TD[sk + i,sj+np.arange(Asize), sj + i-1] += dpdGLeft * exogTrans[k,j]

            TD[sk + i - 1,sj+np.arange(Asize), sj + i ] += -dpdGRight * exogTrans[k,j]
            TD[sk + i - 1,sj+np.arange(Asize), sj + i-1] += -dpdGLeft * exogTrans[k,j]

    assert np.allclose(TD.sum(axis=0), np.zeros(N*Asize))
    return TD


def Deriv_wealthResidual_G(ans,G,D_L,D):
    J = -(Deriv_MakeTransMat(G.reshape(N,Asize)) * D_L.reshape(1,N*Asize,1)).sum(axis = 1)
    J = J[1:] # drop first equation
    return lambda g : g  @ J

def Deriv_wealthResidual_D(ans,G,D_L,D):
    J = np.eye(len(D))
    J = J[1:] # drop first equation
    return lambda g : g  @ J

def Deriv_wealthResidual_D_L(ans,G,D_L,D):
    J = -MakeTransMat(G.reshape(N,Asize))
    J = J[1:] # drop first equation
    return lambda g : g  @ J

defvjp(wealthResidual,Deriv_wealthResidual_G,Deriv_wealthResidual_D_L,Deriv_wealthResidual_D)


# Linearize
print("A")
AMat = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
print("B")
BMat = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
print("C")
CMat = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
print("E")
EMat = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)
EMat = EMat[:,np.newaxis]  # this is needed when we have just one shock so that Emat is a column vector


P, Q = solve(AMat,BMat,CMat,EMat)



# Calculate an impulse response for aggregate variables
IRF_Reiter = np.zeros((2,100))
X = Q * 0.01
for t in range(0,100):
    IRF_Reiter[:,[t]] = X[-2:]
    X = P @ X


plt.figure()
lnReiter, = plt.plot(IRF_Reiter[0,:], label = 'Het. agents')
lnRBC, = plt.plot(IRF_RBC[2,:], label = 'RBC')
plt.legend(handles = [lnReiter,lnRBC])
plt.title('IRF for Capital to TFP shock')
plt.show()
