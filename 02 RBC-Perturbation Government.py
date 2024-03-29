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
nX = 9
# Number of shocks
nEps = 1
# Indexing the variables
iZ, iY, iC, iI, iG, iR, iK, iW, iL = range(nX)


# Parameters
alpha = 0.4
beta = 0.99
gamma = 2
vega = 0.36
delta = 0.019
rho = 0.95
rho_g = 0.9
omega = 0.2
sigma_z = np.sqrt(0.000049)
sigma_g = np.sqrt(0.000009)

# Defining a function, which gives back the steady state
def SteadyState():
    Z = 1.
    R = 1/beta
    W = (1-alpha)*((alpha*Z)/(R-(1-delta)))**(alpha/(1-alpha))
    KL = ((R-1+delta)/alpha)**(1./(alpha-1))
    YL = KL**alpha
    CL = (1-omega)*YL - delta*KL
    Ll = (1-vega)/vega*CL/(1-alpha)*KL**(-alpha)
    L = 1/(1+Ll)
    K = KL*L
    Y = YL*L
    G = 0.2*Y
    C = CL*L
    I = Y - C - G

    X = np.zeros(nX)
    X[[iZ, iY, iC, iI, iG, iR, iK, iW, iL]] = (Z, Y, C, I, G, R, K,  W, L)
    return X


# Get the steady state
X_SS = SteadyState()
X_EXP = np.array(("Prod.", "Output", "Consumption", "Investment", "Government Exp.", "Interest", "Capital", "Wage", "Labour", ))
epsilon_SS = np.zeros(2)
print("Variables: {}".format(X_EXP))
print("Steady state: {}".format(X_SS))


# Model equations
def F(X_Lag,X,X_Prime,epsilon,X_SS):

    # Unpack
    epsilon_z, epsilon_g = epsilon
    Z_SS, Y_SS, C_SS, I_SS, G_SS, R_SS, K_SS, W_SS, L_SS = X_SS
    Z, Y, C, I, G, R, K, W, L = X
    Z_L, Y_L, C_L, I_L, G_L, R_L, K_L, W_L, L_L = X_Lag
    Z_P, Y_P, C_P, I_P, G_P, R_P, K_P, W_P, L_P = X_Prime
    return np.hstack((
            beta * R_P * vega/C_P*(C_P**vega*(1-L_P)**(1-vega))**(1-gamma) / 
            (vega/C*(C**vega*(1-L)**(1-vega))**(1-gamma)) - 1.0,           # Euler equation
            alpha * Z  * (K_L/L) **(alpha-1) + 1 -delta - R,               # MPK
            (1-alpha)*Z*(K_L/L)**(alpha) - W,                              # MPL 
            C/(1-L) - vega/(1-vega)*(1-alpha)*Z*(K_L/L)**alpha,            # Labour allocation
            Y - C - G - I,                                                 # Aggregate resource constraint
            Z*K_L**alpha * (L)**(1-alpha) - Y,                             # Production function
            (1-delta) * K_L + I - K,                                       # Investment
            rho * np.log(Z_L) + epsilon_z - np.log(Z),                     # TFP evolution
            rho_g * np.log(G_L) + (1-rho_g)*np.log(omega*Y_SS) + epsilon_g - np.log(G) # Law of motion for G
            ))


# Check whether at the steady state F is zero
print(F(X_SS,X_SS,X_SS,epsilon_SS, X_SS))
assert(np.allclose( F(X_SS,X_SS,X_SS,epsilon_SS, X_SS) , np.zeros(nX)))


# Compute the numerical derivative
A = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS, X_SS))(X_SS)
B = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS, X_SS))(X_SS)
C = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS, X_SS))(X_SS)
E = jacobian(lambda x: F(X_SS,X_SS,X_SS,x, X_SS))(epsilon_SS)


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
T = 200
IRF_RBC = np.zeros((nX,T))
IRF_RBC[:,0] = np.dot(Q, np.array((0,0.01)))


# Impulse response functions for 100 periods
for t in range(1,T):
    IRF_RBC[:,t] = P@IRF_RBC[:,t-1]


# Normalizing with respect to the steady state
for i in range(nX):
    IRF_RBC[i,:] = IRF_RBC[i,:] / X_SS[i] * 100
# Normalizing the interest rate into percentage points difference
IRF_RBC[1] = IRF_RBC[1] * X_SS[1] 


# List with the variable names
names = ["TFP", "Output", "Consumption", "Investment", "Government Exp.", "Interest", "Capital", "Wage", "Labour"]


# Plotting the results of the IRF
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (10,5))
for i in range(nX):
    row = i // 3
    col = i % 3
    axes[row, col].plot(IRF_RBC[i,:])
    axes[row, col].plot(np.zeros(T))
    axes[row, col].set_title(names[i])
fig.tight_layout()
#plt.show()


# Comparison of the volatility of real variables and the model variables
T = 50000
TT = 500 # Periods that are plotted in the end
# Defining empty matrices for simulation and drawing shocks
SIM_RBC = np.zeros((nX,T))
mean = np.array([0,0])
cov = np.array([[sigma_z,-0.0005],[-0.0005, sigma_g]])
eps_t = np.random.multivariate_normal(mean,cov,T)

# Calculating the intercept for the simulation
intercept = (np.eye(nX) - P)@X_SS
# Initialize the variables at their steady state
SIM_RBC[:,0] = X_SS
for t in range(1,T):
    # Development of individual variables
    SIM_RBC[:,t] = intercept + P@SIM_RBC[:,t-1] + Q@eps_t[t]
    # Transition of shock in logs, first is TFP, second is Gov. Exp.
    SIM_RBC[0,t] = np.exp(P[0,0]*np.log(SIM_RBC[0,t-1]) + Q[0]@eps_t[t,:])
    SIM_RBC[4,t] = np.exp(P[4,4]*np.log(SIM_RBC[4,t-1]) + (1-rho_g)*np.log(omega*X_SS[iY]) + Q[4]@eps_t[t,:])


# Plotting the development
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (10,5))
for i in range(nX):
    row = i // 3
    col = i % 3
    axes[row, col].plot(SIM_RBC[i,0:TT])
    axes[row, col].plot(np.ones(TT)*X_SS[i])
    axes[row, col].set_title(names[i])
fig.tight_layout()
plt.show()


# Quickly renaming for easier reference
Y = SIM_RBC


# Print the results of the simulation
print("\nThe stochastic steady state is %F, with the true being %F" % (np.mean(Y[iK,:]), X_SS[iK]))
print("The volatility of output, consumption and investment are %F, %F, and %F." % (np.std(Y[iY])*100/np.mean(Y[iY]),np.std(Y[iC])*100/np.mean(Y[iC]), np.std(Y[iI])*100/np.mean(Y[iI])))
print("The mean of consumption, investment, capital, and labor in relation to output are %F, %F, %F, and %F." % (np.mean(Y[iC]*100/Y[iY]), np.mean(Y[iI]*100/Y[iY]), np.mean(Y[iK]*100/(4*Y[iY])), np.mean(Y[iL]*100)))
print("The CV of consumption, investment and labor in relation to the CV of output are %F, %F, and %F." % ((np.std(Y[iC])*100/np.mean(Y[iC]))/(np.std(Y[iY])*100/np.mean(Y[iY])),(np.std(Y[iI])*100/np.mean(Y[iI]))/(np.std(Y[iY])*100/np.mean(Y[iY])),(np.std(Y[iL])*100/np.mean(Y[iL]))/(np.std(Y[iY])*100/np.mean(Y[iY]))))
print("The correlation of consumption, investment and labor with output are %F, %F, and %F." %(np.corrcoef(Y[iY],Y[iC])[0,1], np.corrcoef(Y[iY],Y[iI])[0,1], np.corrcoef(Y[iY], Y[iL])[0,1]))

