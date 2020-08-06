import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)

from Support import SolveSystem

# Indexing
nX = 5
nEps = 1
iZ, iR, iK, iY, iC = range(nX)


# Parameters
alpha = 0.4
beta = 0.99
gamma = 2.0
delta = 0.019
rho = 0.95




def SteadyState():
    Z = 1.
    R = 1/beta
    K = ((R-1+delta)/alpha)**(1./(alpha-1))
    Y = K**alpha
    C = Y - delta*K

    X = np.zeros(nX)
    X[[iZ, iR, iK, iY, iC]] = (Z, R, K, Y, C)
    return X

X_SS = SteadyState()
epsilon_SS = 0.0
print("Steady state: {}".format(X_SS))

# Model equations
def F(X_Lag,X,X_Prime,epsilon):

    # Unpack
    Z, R, K, Y, C = X
    Z_L, R_L, K_L, Y_L, C_L = X_Lag
    Z_P, R_P, K_P, Y_P, C_P = X_Prime


    return np.hstack((
            beta * R_P * C_P**(-gamma) * C**gamma - 1.0, # Euler equation
            alpha * Z  * K_L **(alpha-1) + 1 -delta - R, # MPK
            (1-delta) * K_L + Y - C - K,# Aggregate resource constraint
            Z * K_L**alpha - Y,# Production function
            rho * np.log(Z_L) + epsilon - np.log(Z)# TFP evolution
            ))


# Check steady state
assert( np.allclose( F(X_SS,X_SS,X_SS,epsilon_SS) , np.zeros(nX)))


# Linearize
A = jacobian(lambda x: F(X_SS,X_SS,x,epsilon_SS))(X_SS)
B = jacobian(lambda x: F(X_SS,x,X_SS,epsilon_SS))(X_SS)
C = jacobian(lambda x: F(x,X_SS,X_SS,epsilon_SS))(X_SS)
E = jacobian(lambda x: F(X_SS,X_SS,X_SS,x))(epsilon_SS)




P, Q = SolveSystem(A,B,C,E)


# Calculate an impulse response
IRF_RBC = np.zeros((nX,100))
IRF_RBC[:,0] = Q * 0.01

for t in range(1,100):
    IRF_RBC[:,t] = P@IRF_RBC[:,t-1]


# This material below runs only if this file is executed as a script
if __name__ == "__main__":


    print("A: {}".format(A))
    print("B: {}".format(B))
    print("C: {}".format(C))
    print("E: {}".format(E))

    import matplotlib.pyplot as plt
    plt.plot(IRF_RBC[iY,:])