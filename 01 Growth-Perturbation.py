"""
Solves the Stochastic growth model using Perturbation
Only slight modification of the McKay script on this topic
"""


import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)
import matplotlib.pyplot as plt
import warnings


# Number of Variables
nX = 6
# Number of shocks
nEps = 1
# Indexing the variables
iZ, iR, iW, iK, iY, iC = range(nX)


# Parameters
alpha = 1/3
beta = 0.99
gamma = 1
delta = 0.025
rho = 0.95


# Defining a function, which gives back the steady state
def SteadyState():
    Z = 1.
    R = 1/beta
    W = (1-alpha)*((alpha*Z)/(R-(1-delta)))**(alpha/(1-alpha))
    K = ((R-1+delta)/alpha)**(1./(alpha-1))
    Y = K**alpha
    C = Y - delta*K

    X = np.zeros(nX)
    X[[iZ, iR, iW, iK, iY, iC]] = (Z, R, W, K, Y, C)
    return X


# Get the steady state
X_SS = SteadyState()
X_EXP = np.array(("Prod.", "Interest", "Wage", "Capital", "Output", "Consumption"))
epsilon_SS = 0.0
print("Variables: {}".format(X_EXP))
print("Steady state: {}".format(X_SS))


# Model equations
def F(X_Lag,X,X_Prime,epsilon):

    # Unpack
    Z, R, W, K, Y, C = X
    Z_L, R_L, W_L, K_L, Y_L, C_L = X_Lag
    Z_P, R_P, W_P, K_P, Y_P, C_P = X_Prime

    return np.hstack((
            beta * R_P * C_P**(-gamma) * C**gamma - 1.0,    # Euler equation
            alpha * Z  * K_L **(alpha-1) + 1 -delta - R,    # MPK
            (1-alpha) * Z * K_L **alpha - W,                # MPL
            (1-delta) * K_L + Y - C - K,                    # Aggregate resource constraint
            Z * K_L**alpha - Y,                             # Production function
            rho * np.log(Z_L) + epsilon - np.log(Z)         # TFP evolution
            ))


# Check whether at the steady state F is zero
assert( np.allclose( F(X_SS,X_SS,X_SS,epsilon_SS) , np.zeros(nX)))


# Linearize around the steady state
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
        if test < 1e-7:
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
T = 200
IRF_RBC = np.zeros((nX,T))
IRF_RBC[:,0] = Q * 0.01


# Impulse response functions for 100 periods
for t in range(1,T):
    IRF_RBC[:,t] = P@IRF_RBC[:,t-1]
    
# Normalizing with respect to the steady state
for i in range(nX):
    IRF_RBC[i,:] = IRF_RBC[i,:] / X_SS[i] * 100
# Normalizing the interest rate into percentage points difference
IRF_RBC[1] = IRF_RBC[1] * X_SS[1]


# List with the variable names
names = ["TFP", "Interest", "Wage", "Capital", "Output", "Consumption"]

# Plotting the results of the IRF
fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (12,6))
for i in range(nX):
    row = i // 3
    col = i % 3
    axes[row, col].plot(IRF_RBC[i,:])
    axes[row, col].plot(np.zeros(T))
    axes[row, col].set_title(names[i])
fig.tight_layout()
# Dropping the empty plot
#fig.delaxes(axes[1][2])
plt.show()
