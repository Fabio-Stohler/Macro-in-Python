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
nX = 4
# Number of shocks
nEps = 2
# Indexing the variables
ib, iR, iPi, iS = range(nX)


# Parameters
alpha = 0.0
beta = 0.99
gamma = 0.0


# Defining a function, which gives back the steady state
def SteadyState():
    B = 10
    R = 1 / beta
    Pi = 1
    S = B * (R / Pi - 1)

    X = np.zeros(nX)
    X[[ib, iR, iPi, iS]] = (B, R, Pi, S)
    return X


# Get the steady state
X_SS = SteadyState()
X_EXP = np.array(("Bonds", "Interest", "Inflation", "Surplus", ))
epsilon_SS = np.zeros(nEps)
print("Variables: {}".format(X_EXP))
print("Steady state: {}".format(X_SS))


# Model equations
def F(X_Lag,X,X_Prime,epsilon):

    # Unpack
    B, R, Pi, S = X
    B_L, R_L, Pi_L, S_L = X_Lag
    B_P, R_P, Pi_P, S_P = X_Prime
    return np.hstack((
            1 / beta - R / Pi_P,                            # Euler equation
            B + S - B_L * R / Pi,                           # Government BC
            S - X_SS[iS] - gamma * (B_L / R_L - X_SS[ib] / X_SS[iR]) - epsilon[0],              # Behavior Government
            R - X_SS[iR] - alpha * beta * (Pi - X_SS[iPi]) - epsilon[1],   # Taylor rule
            ))


# Check whether at the steady state F is zero
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
IRF_RBC[:,0] = Q[:, 1] * 0.01


# Impulse response functions for T periods
for t in range(1,T):
    IRF_RBC[:,t] = P@IRF_RBC[:,t-1]

# Drop all IRFs that are below e**(-15)
criterion = ((np.abs(IRF_RBC) < 10**(-10))) 
IRF_RBC[criterion] = 0.0


## Normalizing with respect to the steady state
#for i in range(nX):
#    IRF_RBC[i,:] = IRF_RBC[i,:] / X_SS[i] * 100


# List with the variable names
names = ["Bonds", "Interest", "Inflation", "Surplus",]


# Plotting the results of the IRF
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10,5))
for i in range(nX):
    row = i // 2
    col = i % 2
    axes[row, col].plot(IRF_RBC[i,:])
    axes[row, col].plot(np.zeros(T))
    axes[row, col].set_title(names[i])
    fig.tight_layout()
plt.show()


## Comparison of the volatility of real variables and the model variables
#sigma = np.sqrt(0.000049)
#T = 5000
#TT = 500 # Periods that are plotted in the end
## Defining empty matrices for simulation and drawing shocks
#SIM_RBC = np.zeros((nX,T))
#eps_t = np.random.normal(0,sigma,T)
## Calculating the intercept for the simulation
#intercept = (np.eye(nX) - P)@X_SS
## Initialize the variables at their steady state
#SIM_RBC[:,0] = X_SS
#for t in range(1,T):
#    # Development of individual variables
#    SIM_RBC[:,t] = intercept + P@SIM_RBC[:,t-1] + eps_t[t]*Q
#    # Transition of shock in logs
#    SIM_RBC[0,t] = np.exp(P[0,0]*np.log(SIM_RBC[0,t-1]) + Q[0] * eps_t[t])
#
#
## Plotting the development
#fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (18,9))
#for i in range(nX):
#    row = i // 4
#    col = i % 4
#    axes[row, col].plot(SIM_RBC[i,0:TT])
#    axes[row, col].plot(np.ones(TT)*X_SS[i])
#    axes[row, col].set_title(names[i])
#fig.tight_layout()
#plt.show()
#
#
## Quickly renaming for easier reference
#Y = SIM_RBC
#
#
## Print the results of the simulation
#print("\nThe stochastic steady state is %F, with the true being %F" % (np.mean(Y[iK,:]), X_SS[iK]))
#print("The volatility of output, consumption and investment are %F, %F, and %F." % (np.std(Y[iY])*100/np.mean(Y[iY]),np.std(Y[iC])*100/np.mean(Y[iC]), np.std(Y[iI])*100/np.mean(Y[iI])))
#print("The mean of consumption, investment, capital, and labor in relation to output are %F, %F, %F, and %F." % (np.mean(Y[iC]*100/Y[iY]), np.mean(Y[iI]*100/Y[iY]), np.mean(Y[iK]*100/(4*Y[iY])), np.mean(Y[iL]*100)))
#print("The CV of consumption, investment and labor in relation to the CV of output are %F, %F, and %F." % ((np.std(Y[iC])*100/np.mean(Y[iC]))/(np.std(Y[iY])*100/np.mean(Y[iY])),(np.std(Y[iI])*100/np.mean(Y[iI]))/(np.std(Y[iY])*100/np.mean(Y[iY])),(np.std(Y[iL])*100/np.mean(Y[iL]))/(np.std(Y[iY])*100/np.mean(Y[iY]))))
#print("The correlation of consumption, investment and labor with output are %F, %F, and %F." %(np.corrcoef(Y[iY],Y[iC])[0,1], np.corrcoef(Y[iY],Y[iI])[0,1], np.corrcoef(Y[iY], Y[iL])[0,1]))
#
#