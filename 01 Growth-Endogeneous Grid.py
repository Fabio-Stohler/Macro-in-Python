
"""

This solves the stochastic growth model with the endogeneous grid method.

The code adapts the McKay code on the endogeneous grid method to the 
stochastic growth model.

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit
from scipy.interpolate import interp2d


# Supress warning
import warnings
warnings.filterwarnings("ignore")


class HH():
    """
    Setups a class containing all necessary information to solve the model.
    """
    
    def __init__(self,theta = 0.4, beta = 0.99, delta = 0.019, sigma = 2, 
                 rho = 0.95, stdz = 0.007, nz = 21, nk = 500, m = 3):
        """Initialize the class with standard parameters"""
        self.theta, self.beta, self.delta = theta, beta, delta
        self.sigma, self.rho, self.stdz = sigma, rho, stdz
        self.nz, self.nk, self.m = nz, nk, m
        
        #steady state quantities
        self.k_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
        self.c_ss  = self.k_ss*(1-beta*(1-delta)-beta*theta*delta)/(beta*theta)
        
        #discretizing the k grid
        self.kmin = 0.75*self.k_ss
        self.kmax = 1.25*self.k_ss
        self.k = np.linspace(self.kmin,self.kmax,self.nk)
        
        # Approximations
        self.mc = qe.markov.approximation.tauchen(self.rho,self.stdz,0,self.m,
                                                  self.nz)
        self.P = self.mc.P
        self.zs = np.exp(self.mc.state_values)
        
        # Prices
        capital, prod = np.meshgrid(self.k, self.zs)
        self.R = theta*capital**(theta-1)*prod + (1-delta)
        self.w = (1-theta)*capital**(theta)*prod
        
    def utility(self, c):
        """Utility function, dependent on consumption c and sigma"""
        if self.sigma == 1:
            util = np.log(c)
        else:
            util = c**(1-self.sigma)/(1-self.sigma)
        return util
    
    def uprime(self, c):
        """Marginal utility, dependent on consumption and sigma"""
        return c**(-self.sigma)
    
    def uprime_inv(self, mu):
        """Inverse of the marginal utility, 
           dependent on consumption, and sigma"""
        return mu**(-1.0/self.sigma)        


# Generating a HH class:
hh = HH(nz = 21, nk=500)
mc, P, zs = hh.mc, hh.P, hh.zs


# Extracting parameters
nz, nk, beta, sigma = hh.nz, hh.nk, hh.beta, hh.sigma
k, zs = hh.k, hh.zs
tiledGrid = np.tile(k,(nz,1))


# Interpolation function
def interp(x,y,x1):
    N = len(x)
    # Searching for the position in the grid where x1 belongs into
    i = np.minimum(np.maximum(np.searchsorted(x,x1,side='right'),1),N-1)
    # Defining the values which we use for the linear interpolation
    xl = x[i-1]
    xr = x[i]
    yl = y[i-1]
    yr = y[i]
    # Actual interpolation
    y1 = yl + (yr-yl)/(xr-xl) * (x1-xl)
    above = x1 > x[-1]
    below = x1 < x[0]
    # Where x1 is above the highest x, give back a modified interpolation
    y1 = np.where(above,y[-1] +   (x1 - x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), y1)
    # Where x1 is below, give back y[0] and else the interpolated value
    #y1 = np.where(below,y[0],y1)
    y1 = np.where(below, y[0] + (x1 - x[0]) * (y[1] - y[0])/(x[1] - x[0]), y1)
    return y1, i


# Class with prices
interest = hh.R
wage = hh.w


# Given endogeneous assets, get back consumption
def get_c(G,CurrentAssets = tiledGrid):
    """Function returning the vector of consumption for a given policy 
       function G, which we interpolate onto the exogeneous grid"""       
    return np.vstack([interest[i,:]*CurrentAssets[i]+wage[i,:]-
                      interp(G[i],k,CurrentAssets[i])[0] for i in range(nz)])


# Function doing one iteration of the EGM algorithm
def eulerBack(G,Pr,Pr_P,HH):
    """Function taking as input the savings rule G defined on a', prices Pr 
       this and Pr_P next period, as well as a household class HH.
       1. Extracts the consumption this period cp
       2. Calculates marginal utility upcp
       3. Computes the expected value and discounts it to this period upc
       4. From FOC gets optimal consumption allocation
       5. Get's the endogeneous grid in this period"""
    # compute next period's consumption conditional on next period's income
    cp = get_c(G)
    upcp = HH.uprime(cp)
    #compute E(u'(cp))
    Eupcp = np.dot(HH.P,upcp)
    
    #use upc = R' *  beta*Eupcp to solve for upc
    upc = beta*Pr_P.R*Eupcp

    #invert uprime to solve for c
    c = HH.uprime_inv(upc)

    #use budget constraint to find previous assets
    # (a' + c - y)/R = a
    a = (tiledGrid + c - Pr.w)/ Pr.R
    return a, c


def SolveEGM(G,Pr,HH, tol = 1e-8):
    """Solves the households problem with the complete EGM algorithm."""
    test = True
    for it in range(10000):
        a, c = eulerBack(G,Pr,Pr,HH)
        
        if it % 50 == 0:
            test = np.abs(a-G)/(np.abs(a)+np.abs(G)+tol)
            print("it = {0}, test = {1}".format(it,test.max()))
            if np.all(test  < tol):
                break
        G = a
    return G, c


#initialize policy function
G = 10+0.1*tiledGrid


# Running the function
start = time.time()
G, C = SolveEGM(G,hh,hh)
stop = time.time()
print("\nSolving the model with EGM took %F seconds" % (stop - start))


G = np.vstack([interp(G[i],k,tiledGrid[i])[0] for i in range(nz)])


# Function, finding the nearest neighbor
@jit
def find_nearest(array, value):
    """Function, finding the nearest element to value in the array "array" """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Finding the indicator function for VFI
indk = np.copy(G)
for ik in range(nk):
    for iz in range(nz):
        nearestk = find_nearest(k, G[iz,ik])
        indk[iz,ik] = np.where(k == nearestk)[0]


# Extracting the value function to G
def Value(C,indk, maxiter = 1000, tol = 10**(-8)):
    start = time.time()
    error = 1
    iter = 0
    V = hh.utility(C)/(1-hh.beta)
    Vnew = np.copy(V)
    test1 = (error > tol)
    test2 = (iter < maxiter)
    while test1 and test2:
        iter += 1
        for i in range(nz):
            for j in range(nk):
                Vnew[i,j] = hh.utility(C[i,j]) + beta*np.dot(hh.P[i,:], V[:,int(indk[i,j])])
        error = np.linalg.norm(Vnew-V)
        V = np.copy(Vnew)
        test1 = (error > 10**(-8))
        test2 = (iter < maxiter)
    stop = time.time()
    print("Extracting the associated value function took %F seconds." %(stop - start))
    return V


# Getting the value function
V = Value(C,indk)



# Plotting the solution of the households problem              
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,G.transpose())
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
plt.show()


# Simulate the economy
T = 5000
A = hh.mc.simulate(T, init = mc.state_values[int((hh.nz-1)/2)])
A = np.exp(A)
K = np.zeros(T)
Kind = hh.nk/2
K[0] = hh.k[int(Kind)]
g = interp2d(hh.k,hh.zs,G)
sigma = interp2d(hh.k,hh.zs,C)


# Simulating the economy period by period
for t in range(1,T):
    K[t] = g(K[t-1],A[t-1])
out = A*K**hh.theta
cons = np.copy(out)
for t in range(T):
    cons[t] = out[t] - g(K[t],A[t]) + (1-hh.delta)*K[t]
inv = out - cons


# Plot the development of the economy
t = range(T)
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))

axes[0].plot(t, K)
axes[0].set_title("Trajectory of capital")
axes[0].set_xlabel("Period")
axes[0].set_ylabel("Capital")

axes[1].plot(t, out, label = "Output")
axes[1].plot(t, cons, label = "Consumption")
axes[1].plot(t, inv, label = "Investment")
axes[1].set_title("GDP components")
axes[1].set_xlabel("Period")
axes[1].set_ylabel("GDP components")
axes[1].legend(loc=5)
plt.show()


# Printing results of the simulation
print("\nThe stochastic steady state is %F, with the true being %F" 
      % (np.mean(K), hh.k_ss))
print("\nThe volatility of output, consumption and investment are %F, %F, and %F." 
      % (np.std(out)*100/np.mean(out),np.std(cons)*100/np.mean(cons), 
         np.std(inv)*100/np.mean(inv)))

