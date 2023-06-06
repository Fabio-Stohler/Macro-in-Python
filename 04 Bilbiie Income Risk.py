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
from prettytable import PrettyTable


# Number of Variables
nX = 15
# Number of shocks
nEps = 2
# Indexing the variables
iCs, iCh, iC, iNs, iNh, iN, iY, iPi, iD, iW, iI, iL, iS, iXI, iXS = range(nX)


# Parameters
beta = 0.99
sigma = 2.0
varphi = 1.0
eta = 11
psi = 500
tauD = 0.0
tauS = eta / (eta - 1.0) - 1.0
s = 0.98
h = 0.92
rhoI = 0.75
rhoS = 0.5
ps_Y = 1.0
phi = 1.5

# Defining the desired inequality
Gamma = 1.05

# Defining a function, which gives back the steady state
def SteadyState():
    Ch = 1.0
    Cs = Ch / Gamma
    W = 1.0
    L = (1 - s) / (2 - s - h)
    C = L * Ch + (1-L) * Cs
    Ns = (W * Cs ** (-sigma)) ** (1 / varphi)
    Nh = (W * Ch ** (-sigma)) ** (1 / varphi)
    N = L * Nh + (1-L) * Ns
    Pi = 1.0
    Y = C
    D = (1+tauS) * Y - W * N - tauS * Y
    S = s
    I = 1.0 / (beta * (S + (1 - S) * Gamma ** (sigma))) * Pi ** phi
    XI = 1.0
    XS = 1.0

    X = np.zeros(nX)
    X[[iCs, iCh, iC, iNs, iNh, iN, iY, iPi, iD, iW, iI, iL, iS, iXI, iXS]] = (Cs, Ch, C, Ns, Nh, N, Y, Pi, D, W, I, L, S, XI, XS)
    return X

# Adjusting the necessary parameters to incorporate inequality



# Get the steady state
X_SS = SteadyState()
table.add_column("Variables", ["CS", "CH", "C", "Ns", "Nh", "N", "Y", "Pi", "D", "W", "I", "L", "S", "Shock I", "Shock S"])
epsilon_SS = np.zeros(2)
table.add_column("Values", np.round(X_SS, 4))
print(" ")
print(table)
# print("Variables: {}".format(X_EXP))
# print("Steady state: {}".format(X_SS))


# Model equations
def F(X_Lag,X,X_Prime,epsilon,XSS):

    # Unpack
    epsilon_i, epsilon_s = epsilon
    Cs, Ch, C, Ns, Nh, N, Y, Pi, D, W, I, L, S, XI, XS = X
    Cs_L, Ch_L, C_L, Ns_L, Nh_L, N_L, Y_L, Pi_L, D_L, W_L, I_L, L_L, S_L, XI_L, XS_L = X_Lag
    Cs_P, Ch_P, C_P, Ns_P, Nh_P, N_P, Y_P, Pi_P, D_P, W_P, I_P, L_P, S_P, XI_P, XS_P = X_Prime
    Cs_SS, Ch_SS, C_SS, Ns_SS, Nh_SS, N_SS, Y_SS, Pi_SS, D_SS, W_SS, I_SS, L_SS, S_SS, XI_SS, XS_SS = XSS
    return np.hstack((
                # Shocks
                XI - XI_L ** rhoI * np.exp(epsilon_i), # Transition of MP shock
                XS - XS_L ** rhoS * np.exp(epsilon_s), # Transition of risk shock

                # Household
                S / s - XS * (Y / Y_SS) ** ps_Y, # Idiosyncratic risk
                Cs ** (-sigma) - beta * I / Pi_P * (S_P * Cs_P ** (-sigma) + (1 - S_P) * Ch_P ** (-sigma)), # Euler equation
                Cs - W * Ns - (1 - tauD) / (1 - L) * D, # BC of saver
                Ch - W * Nh - tauD / L * D, # BC of HtM household
                Nh ** varphi - W * Ch ** (-sigma), # Labor supply of HtM household
                Ns ** varphi - W * Cs ** (-sigma), # Labor supply of Saver household

                # Distributional changes
                L - h * L_L - (1 - S) * (1 - L_L), # Distribution changes over time

                # Aggregation
                C - L * Ch - (1-L) * Cs, # Aggregate consumption
                N - L * Nh - (1-L) * Ns, # Aggregate labor supply
                #C - (1 - psi / 2 * (Pi - 1.0) ** 2) * Y, # Goods market clearing

                # Firms
                Y - N, # Production function
                D - (1 + tauS) * Y + W * N + tauS * Y, # Profits
                (Pi - 1.0) * Pi - beta * ((Cs_P / Cs) ** (-sigma) * Y / Y_P * (Pi_P - 1.0) * Pi_P) - eta / psi * (W - 1 / (1 / (1 + tauS) * eta / (eta - 1))), # Phillips curve
                
                # Monetary policy
                I - 1 / beta * Pi ** phi * XI # Taylor rule
            ))


# Check whether at the steady state F is zero
print(F(X_SS,X_SS,X_SS,epsilon_SS, X_SS))
assert(np.allclose( F(X_SS,X_SS,X_SS,epsilon_SS, X_SS) , np.zeros(nX)))


# Compute the numerical derivative
A = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS,X_SS))(X_SS)
B = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS,X_SS))(X_SS)
C = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS,X_SS))(X_SS)
E = jacobian(lambda x: F(X_SS,X_SS,X_SS,x,X_SS))(epsilon_SS)


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
T = 15
IRF_MP = np.zeros((nX,T))
IRF_S = np.copy(IRF_MP)
# First shock is monetary, second is risk
shockMP = np.array((0.01, 0.0))
shockS = np.array((0.0, -0.05))
IRF_MP[:,0] = np.transpose(Q @ shockMP)
IRF_S[:,0] = np.transpose(Q @ shockS) 

# Impulse response functions for 100 periods
for t in range(1,T):
    IRF_MP[:,t] = P @ IRF_MP[:,t-1]
    IRF_S[:,t] = P @ IRF_S[:,t-1]

# Drop all IRFs that are below e**(-15)
criterion_MP = ((np.abs(IRF_MP) < 10**(-10)))
criterion_S = ((np.abs(IRF_S) < 10**(-10)))
IRF_MP[criterion_MP] = 0.0
IRF_S[criterion_S] = 0.0

# Dividend is zero in the steady state
floors = [ f for f in range(nX) if f != 8 ]

# Normalizing with respect to the steady state
for i in floors:
    IRF_MP[i,:] = IRF_MP[i,:] / X_SS[i] * 100
    IRF_S[i,:] = IRF_S[i,:] / X_SS[i] * 100


# List with the variable names
names = ["CS", "CH", "C", "Ns", "Nh", "N", "Y", "Pi", "D", "W", "I", "L", "S", "XI", "XS"]


# Plotting the results of the IRF to a MP shock
fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (10,6))
for i in range(nX):
    row = i // 3        # Ganzahlige Division
    col = i % 3         # Rest
    axes[row, col].plot(IRF_MP[i,:])
    axes[row, col].plot(np.zeros(T))
    axes[row, col].set_title(names[i])
fig.tight_layout()
plt.show()

# Plotting the results of the IRF to a S shock
# fig, axes = plt.subplots(nrows = 5, ncols = 3, figsize = (10,6))
# for i in range(nX):
    # row = i // 3        # Ganzahlige Division
    # col = i % 3         # Rest
    # axes[row, col].plot(IRF_S[i,:])
    # axes[row, col].plot(np.zeros(T))
    # axes[row, col].set_title(names[i])
# fig.tight_layout()
# plt.show()