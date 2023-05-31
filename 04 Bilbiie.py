"""
Solves the classic RBC model using Perturbation
Extension of the McKay material with respect to plotting and statistics
Reference: https://alisdairmckay.com/Notes/HetAgents/index.html
"""


import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)
import matplotlib.pyplot as plt
import warnings



# Number of Variables
nX = 12
# Number of shocks
nEps = 1
# Indexing the variables
iCs, iCh, iC, iNs, iNh, iN, iY, iPi, iD, iW, iI, iR = range(nX)


# Parameters
beta = 0.99
sigma = 2
varphi = 1.0
eta = 11
psi = 500
tauD = 0
tauS = eta / (eta - 1.0) - 1.0
s = 0.98
lambdas = 0.22
phi = 1.5


# Defining a function, which gives back the steady state
def SteadyState():
    Cs = 1.0
    Ch = 1.0
    C = lambdas * Ch + (1-lambdas) * Cs
    W = 1.0
    Ns = (W * Cs ** (-sigma)) ** (1 / varphi)
    Nh = (W * Ch ** (-sigma)) ** (1 / varphi)
    N = lambdas * Nh + (1-lambdas) * Ns
    Pi = 1.0
    Y = C
    D = (1+tauS) * Y - W * N - tauS * Y
    I = 1.0 / beta * Pi ** phi
    R = I / Pi

    X = np.zeros(nX)
    X[[iCs, iCh, iC, iNs, iNh, iN, iY, iPi, iD, iW, iI, iR]] = (Cs, Ch, C, Ns, Nh, N, Y, Pi, D, W, I, R)
    return X


# Get the steady state
X_SS = SteadyState()
X_EXP = np.array(("CS", "CH", "C", "Ns", "Nh", "N", "Y", "Pi", "D", "W", "I", "P", "R"))
epsilon_SS = np.zeros(1)
print("Variables: {}".format(X_EXP))
print("Steady state: {}".format(X_SS))


# Model equations
def F(X_Lag,X,X_Prime,epsilon):

    # Unpack
    epsilon_m = epsilon
    Cs, Ch, C, Ns, Nh, N, Y, Pi, D, W, I, R = X
    Cs_L, Ch_L, C_L, Ns_L, Nh_L, N_L, Y_L, Pi_L, D_L, W_L, I_L, R_L = X_Lag
    Cs_P, Ch_P, C_P, Ns_P, Nh_P, N_P, Y_P, Pi_P, D_P, W_P, I_P, R_P = X_Prime
    return np.hstack((
                Cs ** (-sigma) - beta * R * (s * Cs_P ** (-sigma) + (1 - s) * Ch_P ** (-sigma)), # Euler equation
                Ch - W * Nh - tauD / lambdas * D, # BC of HtM household
                Nh ** varphi - W * Ch ** (-sigma), # Labor supply of HtM household
                Ns ** varphi - W * Cs ** (-sigma), # Labor supply of Saver household
                C - lambdas * Ch - (1-lambdas) * Cs, # Aggregate consumption
                N - lambdas * Nh - (1-lambdas) * Ns, # Aggregate labor supply
                D - (1 + tauS) * Y + W * N + tauS * Y, # Profits
                (Pi - 1.0) * Pi - beta * ((Cs_P / Cs) ** (-sigma) * Y / Y_P * (Pi_P - 1.0) * Pi_P) - eta / psi * (W - 1 / (1 / (1 + tauS) * eta / (eta - 1))), # Phillips curve
                Y - N, # Production function
                W - 1.0, # Wage setting
                I - 1 / beta * Pi ** phi * np.exp(epsilon_m), # Taylor rule
                R - I / Pi_P # Fisher equation
            ))


# Check whether at the steady state F is zero
print(F(X_SS,X_SS,X_SS,epsilon_SS))
assert(np.allclose( F(X_SS,X_SS,X_SS,epsilon_SS) , np.zeros(nX)))


# Compute the numerical derivative
A = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
B = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
C = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
E = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)


# Function to solve the system based on McKays material
def SolveSystem(A,B,C,E,P0=None):
    # Solve the system using linear time iteration as in Rendahl (2017)
    #print("Solving the system")
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
        #if it % 20 == 0:
            #print(test)
        if test < 1e-10:
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


# Using the function to solve the system
P, Q = SolveSystem(A,B,C,E)


# Calculate an impulse response
T = 20
IRF_RBC = np.zeros((nX,T))
IRF_RBC[:,0] = np.dot(Q, np.array(0.01))[:,0]


# Impulse response functions for 100 periods
for t in range(1,T):
    IRF_RBC[:,t] = P@IRF_RBC[:,t-1]


# Normalizing with respect to the steady state
floors = [ f for f in range(nX) if f != 8 ]
for i in floors:
    IRF_RBC[i,:] = IRF_RBC[i,:] / X_SS[i] * 100


# List with the variable names
names = ["CS", "CH", "C", "Ns", "Nh", "N", "Y", "Pi", "D", "W", "I", "R"]


# Plotting the results of the IRF
fig, axes = plt.subplots(nrows = 4, ncols = 3, figsize = (10,5))
for i in range(nX):
    row = i // 3        # Ganzahlige Division
    col = i % 3         # Rest
    axes[row, col].plot(IRF_RBC[i,:])
    axes[row, col].plot(np.zeros(T))
    axes[row, col].set_title(names[i])
fig.tight_layout()
plt.show()

