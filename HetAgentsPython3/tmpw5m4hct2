import autograd.numpy as np
from autograd import jacobian
import matplotlib.pyplot as plt 

np.set_printoptions(suppress=True,precision=4)

from Support import SolveSystem


# Parameters and setup
beta = 0.995
gamma = 2.0
rho = 0.95
rho_xi = 0.80
zeta = 0.8
psi = 0.1
mu = 1.2  # markup  = epsilon / (epsilon - 1)
mu_epsilon = mu/(mu-1)
theta = 0.75
omega = 1.5
ubar = 0.06
delta = 0.15 # job separation prob
Mbar = (1-ubar)*delta/(ubar+delta*(1-ubar))
wbar = 1/mu - delta * psi * Mbar
B = 0.6
ben = wbar * 0.5


def u(c):
    return (c**(1-gamma)-1.0)/(1.0-gamma)

def uPrime(c):
    return c**(-gamma)

def uPrimeInv(up):
    return up**(-1.0/gamma)


Rstar = 1/beta


def AggResidual(C,C_P, u, u_L, R, R_P, i_L, i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P, Y, Z_L,Z, xi_L,xi,epsilon):
    #               1   2  3   4   5   6  7  8 9 10 11
    # Equations for u , R, M, pi, pA, pB, Z  C xi Y i
    H = 1-u - (1-delta)*(1-u_L)
    marg_cost = (wbar * (M/Mbar)**zeta + psi *M  - (1-delta)*psi *M_P)/ Z
    # C = Agg_C(D_L,Pr,u)
    return np.hstack((Y-C-psi*M*H,  #   agg res constraint
                      1+i - Rstar * pi**omega * xi,         #   mon pol rule
                      R - (1+i_L)/pi,
                      M - (1-u-(1-delta)*(1-u_L))/(u_L + delta*(1-u_L)),  #  labor market dynamics
                      pi - theta**(1./(1-mu_epsilon))*(1-(1-theta)*(pA/pB)**(1-mu_epsilon))**(1./(mu_epsilon-1)), #  inflation
                      -pA + mu * Y * marg_cost + theta * pi_P**mu_epsilon * pA_P / R, #  aux inflation equ 1
                      -pB + Y  + theta * pi_P**(mu_epsilon-1) * pB_P / R, #  aux inflation equ 2
                      np.log(Z) - rho*np.log(Z_L)-epsilon[0] ,   #  TFP evolution
                      np.log(xi) - rho_xi*np.log(xi_L)-epsilon[1], # monetary shock evolution
                      -uPrime(C) + beta * R_P * uPrime(C_P),
                      -Y + Z * (1-u)))



def F(X_L,X,X_P,epsilon):
    # Bundle the equations of the model

    # Step 1: unpack


    u_L, R_L, M_L, pi_L, pA_L, pB_L, Z_L, xi_L, C_L, Y_L, i_L = X_L
    u, R, M, pi, pA, pB, Z, xi, C, Y, i = X
    u_P, R_P, M_P, pi_P, pA_P, pB_P, Z_P, xi_P, C_P, Y_P, i_P = X_P



    # Step 3: bundle equations
    return AggResidual(C, C_P, u, u_L, R, R_P, i_L, i, M, M_P, pi,pi_P,pA,pB, pA_P,pB_P,Y, Z_L,Z,xi_L,xi,epsilon)


# Assemble steady state variables
pB  = (1-ubar)  /(1-theta/Rstar)
C = (1-ubar) - psi * Mbar * delta * (1-ubar)
Agg_SS = np.array((ubar,Rstar,Mbar,1.0,pB,pB,1.0,1.0,C,1-ubar,Rstar-1))

X_SS = Agg_SS
epsilon_SS = np.zeros(2)

# test steady state
assert np.allclose(F(X_SS,X_SS,X_SS,epsilon_SS) , np.zeros(Agg_SS.size))


# Linearize
print("A")
AMat = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
print("B")
BMat = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
print("C")
CMat = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
print("E")
EMat = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)


P, Q = SolveSystem(AMat,BMat,CMat,EMat)



# Calculate an impulse response
T = 100
IRF = np.zeros((X_SS.size,T))
IRF[:,0] = Q[:,1]  # Shock of size epsilon= 1 is implicit here

for t in range(1,T):
    IRF[:,t] = P@IRF[:,t-1]

for i in range(11):
    plt.plot(IRF[i,:])
    plt.show()
