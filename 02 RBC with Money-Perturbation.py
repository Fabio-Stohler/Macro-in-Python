"""
Solves the RBC model with money and without capital using Perturbation
Extension of the McKay material with respect to plotting and statistics
Reference: https://alisdairmckay.com/Notes/HetAgents/index.html
"""


import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)
import matplotlib.pyplot as plt
import warnings


# Number of Variables
nX = 9
# Number of shocks
nEps = 2
# Indexing the variables
iZ, iY, iC, iW, iI, iM, iPI, iL, iEPS = range(nX)


# Parameters
alpha = 0.4
beta = 0.99
gamma = 1
# Disutility from labor
psi = 2

# Chi of 5 gives approximately 1/3 of hours work in equilibrium
chi = 20

# Utility from money
phi = 2

# Growth rate of money supply
mu = 0.01

# Autocorrelation of the shock to money supply and technology
rho_m = 0.75
rho_a = 0.98

# Defining a function, which gives back the steady state
def SteadyState():
    Z = 1.
    PI = mu
    I = (1+PI)/beta - 1
    L = ((1-alpha)/chi*Z**(gamma-1))**(1/(alpha+gamma*(1-alpha)+psi))
    W = (1-alpha)*Z*L**(-alpha)
    Y = Z*L**(1-alpha)
    C = Y
    M = (C**(-gamma)*I/(1+I))**(-1/phi)
    EPS = 1

    X = np.zeros(nX)
    X[[iZ, iY, iC, iW, iI, iM, iPI, iL, iEPS]] = (Z, Y, C, W, I, M, PI, L, EPS)
    return X


# Get the steady state
X_SS = SteadyState()
X_EXP = np.array(("Prod.", "Output", "Consumption", "Wage", "Interest", "Real Money", "Inflation", "Labour", "Aux", ))
epsilon_SS = np.zeros(nEps)
print("Variables: {}".format(X_EXP))
print("Steady state: {}".format(X_SS))


# Model equations
def F(X_Lag,X,X_Prime,epsilon):
    # Unpack
    Z, Y, C, W, I, M, PI, L, EPS = X
    Z_L, Y_L, C_L, W_L, I_L, M_L, PI_L, L_L, EPS_L = X_Lag
    Z_P, Y_P, C_P, W_P, I_P, M_P, PI_P, L_P, EPS_P = X_Prime
    epsilon_a, epsilon_m = epsilon
    return np.hstack((
            rho_a*np.log(Z_L) + epsilon_a - np.log(Z),                  # TFP evolution
            Z*L**(1-alpha) - Y,                                         # Production function
            Y - C,                                                      # Aggregate resource constraint
            (1-alpha)*Z*L**(-alpha) - W,                                # MPL
            C**(-gamma) - beta*C_P**(-gamma)*(1+I)/(1+PI_P),            # Euler equation
            C**(-gamma)*I/(1+I) - (M)**(-phi),                          # Real Money demand
            np.log(EPS) - rho_m*np.log(EPS_L) - epsilon_m,              # Auxilary equation for epsilon
            M - (1+mu)/(1+PI)*M_L*np.exp(np.log(EPS)),                  # Real Money supply development
            W - chi*L**psi*C**gamma,                                    # Labour allocation            
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

# Calculate an impulse response for a real monetary shock
T = 40
IRF_RBC = np.zeros((nX,T))
shock = np.array((0,0.01))
IRF_RBC[:,0] = np.transpose(Q @ shock)

# Impulse response functions for 100 periods
for t in range(1,T):
    IRF_RBC[:,t] = P@IRF_RBC[:,t-1]


# Normalizing with respect to the steady state
for i in range(nX):
    IRF_RBC[i,:] = IRF_RBC[i,:] / X_SS[i] * 100
# Normalizing the interest rate and inflation into percentage points difference
IRF_RBC[4] = IRF_RBC[4] * X_SS[4]
IRF_RBC[6] = IRF_RBC[6] * X_SS[6]

# Drop all IRFs that are below e**(-15)
criterion = ((np.abs(IRF_RBC) < 10**(-10))) 
IRF_RBC[criterion] = 0.0


# List with the variable names
names = ["Prod.", "Output", "Consumption", "Wage", "Interest", "Money", "Inflation", "Labour"]


# Plotting the results of the IRF
fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (18,9))
for i in range(nX-1):
    row = i // 4
    col = i % 4
    axes[row, col].plot(IRF_RBC[i,:])
    axes[row, col].plot(np.zeros(T))
    axes[row, col].set_title(names[i])
fig.tight_layout()



# Comparison of the volatility of real variables and the model variables
sigma_a = np.sqrt(0.000049)
sigma_m = np.sqrt(0.0001)
T = 5000
TT = 500 # Periods that are plotted in the end

# Defining empty matrices for simulation and drawing shocks
SIM_RBC = np.zeros((nX,T))
eps_a = np.random.normal(0,sigma_a,T)
eps_m = np.random.normal(0,sigma_m,T)
eps_t = np.array((eps_a, eps_m))

# Calculating the intercept for the simulation
intercept = (np.eye(nX) - P)@X_SS

# Initialize the variables at their steady state
SIM_RBC[:,0] = X_SS
for t in range(1,T):
    # Development of individual variables
    SIM_RBC[:,t] = intercept + P@SIM_RBC[:,t-1] + Q@eps_t[:,t]
    # Transition of shock in logs
    SIM_RBC[0,t] = np.exp(P[0,0]*np.log(SIM_RBC[0,t-1]) + eps_t[0,t])
    SIM_RBC[8,t] = np.exp(P[8,8]*np.log(SIM_RBC[8,t-1]) + eps_t[1,t])


# Plotting the development
fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (18,9))
for i in range(nX-1):
    row = i // 4
    col = i % 4
    axes[row, col].plot(SIM_RBC[i,0:TT])
    axes[row, col].plot(np.ones(TT)*X_SS[i])
    axes[row, col].set_title(names[i])
fig.tight_layout()
plt.show()

