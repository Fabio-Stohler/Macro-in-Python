import autograd.numpy as np
from autograd import jacobian
from autograd.extend import defvjp, primitive
np.set_printoptions(suppress=True,precision=4)


from HANK_SS import (A, Asize, Prices, N, interp, MakeTransMat, G, Rstar, D,
                    MakeTransMat_Savings, MakeTransMat_Emp, Agg_Assets)

from HANK_EGM import (get_c, mu_epsilon, wbar, ubar, Mbar, zeta, mu, theta,
                        eulerBack, delta, psi, ben, omega, rho, B, rho_xi)

from SolveQZ import solve

from RANK import IRF as IRFRANK
import matplotlib.pyplot as plt


# note, D tracks end of period states
# For example, aggregate consumption in t should be computed with D_L shuffled
# with MakeTransMat_Emp(Pr): AggC(G, Pr, MakeTransMat_Emp(Pr) @ D_L )


# rewrite eulerBack to deliver residual function
def eulerResidual(G,G_P,Pr,Pr_P):
# The arguments are the savings policy rules in the current and next period
# parameterized as the a values associated with savings grid for a'

    a, c = eulerBack(G_P.reshape(N,Asize),Pr,Pr_P)
    c2 = get_c(G.reshape(N,Asize),Pr,CurrentAssets = a)
    return (c/c2 - 1).reshape(-1)


# get residuals of distribution of wealth equations
@primitive
def wealthResidual(G,D_L,D,M):
    return (D - MakeTransMat(G.reshape(N,Asize),M) @ D_L)[1:]  # drop one redundant equation (prob distribution sums to 1)



def Deriv_MakeTransMat_Savings(G):
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

        TD[sj + i,sj+np.arange(Asize), sj + i ] += dpdGRight
        TD[sj + i,sj+np.arange(Asize), sj + i-1] += dpdGLeft

        TD[sj + i - 1,sj+np.arange(Asize), sj + i ] += -dpdGRight
        TD[sj + i - 1,sj+np.arange(Asize), sj + i-1] += -dpdGLeft

    assert np.allclose(TD.sum(axis=0), np.zeros(N*Asize))

    return TD



def Deriv_wealthResidual_G(ans,G,D_L,D,M):
    J = -(Deriv_MakeTransMat_Savings(G.reshape(N,Asize)) * (MakeTransMat_Emp(M) @ D_L).reshape(1,N*Asize,1)).sum(axis = 1)
    J = J[1:] # drop first equation
    return lambda g : g  @ J

def Deriv_wealthResidual_D(ans,G,D_L,D,M):
    J = np.eye(len(D))
    J = J[1:] # drop first equation
    return lambda g : g  @ J

def Deriv_wealthResidual_D_L(ans,G,D_L,D,M):
    J = -MakeTransMat(G.reshape(N,Asize),M)
    J = J[1:] # drop first equation
    return lambda g : g  @ J

def Deriv_wealthResidual_M(ans,G,D_L,D,M):
    TS = MakeTransMat_Savings(G.reshape(N,Asize))
    DTE = np.array([[-1,-delta],[1,delta]])
    J = []
    for e in range(N):
        ofs = np.array((e,e+1)) * Asize
        Je = 0.0
        for elag in range(N):
            ofslag = elag * Asize + np.arange(Asize)
            Je +=  DTE[e,elag] *  D_L[ofslag]

        J.append(TS[ofs[0]:ofs[1],ofs[0]:ofs[1]] @ Je )

    J = np.hstack(J)
    J = -J[1:] # drop first equation
    return lambda g : g  @ J



defvjp(wealthResidual,Deriv_wealthResidual_G,Deriv_wealthResidual_D_L,Deriv_wealthResidual_D,Deriv_wealthResidual_M)



# def Agg_C(D_L,Pr,u):
#     savings = (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
#     assets = (A.reshape(1,Asize) * D.reshape(N,Asize)).sum()
#     return Pr.RLag * assets + (1-u)*Pr.employedIncome + u * ben - savings

def AggResidual(Pr,D, u, u_L, R,i_L,i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z, xi_L, xi,epsilon):
    #               1   2  3   4   5   6  7 8
    # Equations for u , R, M, pi, pA, pB, Z xi
    Y = Z * (1-u)
    H = 1-u - (1-delta)*(1-u_L)
    marg_cost = (wbar * (M/Mbar)**zeta + psi *M  - (1-delta)*psi *M_P)/ Z
    # C = Agg_C(D_L,Pr,u)
    return np.hstack((Agg_Assets(D) - B,  # 1  Bond clearing
                      1+i - Rstar * pi**omega * xi,         # 2  mon pol rule
                      R - (1+i_L)/pi,
                      M - (1-u-(1-delta)*(1-u_L))/(u_L + delta*(1-u_L)),  # 3 labor market dynamics
                      pi - theta**(1./(1-mu_epsilon))*(1-(1-theta)*(pA/pB)**(1-mu_epsilon))**(1./(mu_epsilon-1)), # 4 inflation
                      -pA + mu * Y * marg_cost + theta * pi_P**mu_epsilon * pA_P / R, # 5 aux inflation equ 1
                      -pB + Y  + theta * pi_P**(mu_epsilon-1) * pB_P / R, # 6 aux inflation equ 2
                      np.log(Z) - rho*np.log(Z_L)-epsilon[0] ,   # 7 TFP evolution
                      np.log(xi) - rho_xi*np.log(xi_L)-epsilon[1])) # monetary shock evolution


def F(X_L,X,X_P,epsilon):
    # Bundle the equations of the model

    # Step 1: unpack
    m = N*Asize
    G_L,D_L,Agg_L = X_L[:m], X_L[m:(2*m-1)], X_L[2*m-1:]
    G  ,D  ,Agg   = X[:m]  , X[m:(2*m-1)],   X[2*m-1:]
    G_P,D_P,Agg_P = X_P[:m], X_P[m:(2*m-1)], X_P[2*m-1:]

    u_L, R_L, i_L, M_L, pi_L, pA_L, pB_L, Z_L, xi_L = Agg_L
    u, R, i, M, pi, pA, pB, Z, xi = Agg
    u_P, R_P, i_P, M_P, pi_P, pA_P, pB_P, Z_P, xi_P = Agg_P

    D_L = np.hstack((1-D_L.sum(), D_L))
    D = np.hstack((1-D.sum(), D))
    D_P = np.hstack((1-D_P.sum(), D_P))

    # Step 2: prices
    Pr = Prices(R,M,Z,u,u_L)
    Pr_P = Prices(R_P,M_P,Z_P,u_P,u)

    # Step 3: bundle equations
    return np.hstack( (eulerResidual(G,G_P,Pr,Pr_P), wealthResidual(G,D_L,D,Pr.M), AggResidual(Pr,D, u, u_L, R, i_L, i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Z_L,Z,xi_L,xi,epsilon) ) )


# Assemble steady state variables
pB  = (1-ubar)  /(1-theta/Rstar)
Agg_SS = np.array((ubar,Rstar,Rstar-1.,Mbar,1.0,pB,pB,1.0,1.0))

X_SS = np.hstack((G.reshape(-1), D[1:], Agg_SS))
epsilon_SS = np.zeros(2)

# test steady state
assert np.allclose(F(X_SS,X_SS,X_SS,epsilon_SS) , np.zeros(2*N*Asize-1+Agg_SS.size))


# Linearize
print("A")
AMat = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
print("B")
BMat = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
print("C")
CMat = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
print("E")
EMat = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)



P, Q = solve(AMat,BMat,CMat,EMat)


# P, Q = SolveSystem(AMat,BMat,CMat,EMat,P0=G1)


# Calculate an impulse response
T = 100
IRF = np.zeros((X_SS.size,T))
IRF[:,0] = Q[:,1]  # Shock of size epsilon= 1 is implicit here


for t in range(1,T):
    IRF[:,t] = P@IRF[:,t-1]


plt.plot(IRF[2*N*Asize-1])
plt.plot(IRFRANK[0])
plt.title("Response of unemployment to monetary shock")
plt.legend(["HANK","RANK"])
plt.xlim([0,20])
plt.ylabel('Percentage points')
plt.xlabel('Quarter')
plt.show()
