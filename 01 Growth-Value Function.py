# -*- coding: utf-8 -*-
"""

This solves the stochastic growth model with value function iteration

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit

# Supress warning
import warnings
warnings.filterwarnings("ignore")


class HH():
    
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
        kmin = 0.8*self.k_ss
        kmax = 1.2*self.k_ss
        self.k = np.linspace(kmin,kmax,self.nk)
        
    def utiltiy(self, c):
        """Utility function, dependent on consumption c and sigma"""
        if self.sigma == 1:
            util = np.log(c)
        else:
            util = c**(1-self.sigma)/(1-self.sigma)
        return util
    
    def markov(self):
        """Approximates the transistion probability of an AR(1) process
           using the methodology of Tauchen (1986) using the quantecon package
           
           Gives back the markov chain, the transition probabilities and the
           respective values of the states.
           """
        self.mc = qe.markov.approximation.tauchen(self.rho,self.stdz,0,self.m,
                                                  self.nz)
        self.P = self.mc.P
        self.zs = np.exp(self.mc.state_values)
        return self.mc, self.P, self.zs


# Generating a HH class:
hh = HH(nk=400)
mc, P, zs = hh.markov()


#Value function iteration function
@jit
def value_function(HH, tolv=10**(-8), maxiter=5000):
    """
    Value function iteration that, given a class of HH solves for the 
    optimal value function, and its associated policy function.
    """
    # Extracting the values
    theta, beta, delta, sigma = HH.theta, HH.beta, HH.delta, HH.sigma
    nz, nk, zs, k, P = HH.nz, HH.nk, HH.zs, HH.k, HH.P
    c_ss = HH.c_ss
    
    start = time.time()
    # Setting up the initial problem
    u    = c_ss**(1-sigma)/(1-sigma)
    V    = np.ones((nz,nk))*u/(1-beta)
    v    = np.zeros((nz,nk))
    g    = np.zeros((nz,nk))
    newV = np.zeros((nz,nk))
    iter    = 0
    diffV   = 10
    # Checking the continuation criteria
    test1 = (iter < maxiter)
    test2 = (diffV > tolv)
    while (test1 and test2):
        iter = iter+1
        for iz in range(0,nz):
            for ik in range(0,nk):
                c = zs[iz]*k[ik]**theta+(1-delta)*k[ik]-k
                u = c**(1-sigma)/(1-sigma)
                u[c<0]= -1000000000
                v = u+beta*(np.dot(P[iz,:],V))
                newV[iz,ik] = max(v)
                ind         = np.argmax(v)
                g[iz,ik]    = k[ind]
        diffV = np.linalg.norm(newV-V)
        V = newV.copy()
        print(iter, diffV)
        test1 = (iter<maxiter)
        test2 = (diffV>tolv)
    stop = time.time() 
    print("\nValue function iteration converged after %.0F iterations and %.5F seconds" % (iter, (stop-start)))
    return V, g


# Running the function
V, g = value_function(hh)
       

#Plotting the value and policy function                 
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,g.transpose())
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Simulate the economy
T = 5000


# Setup the arrays for the later simulation
A = hh.mc.simulate(T, init = mc.state_values[int((hh.nz-1)/2)])
Aind = mc.get_index(A)
A = np.exp(A)
K = np.zeros(T)
Kind = np.copy(K)
Kind[0] = hh.nk/2
K[0] = hh.k[int(Kind[0])]


# Simulating the economy period by period
for t in range(1,T):
    K[t] = g[int(Aind[t-1]),int(Kind[t-1])]
    Kind[t] = np.where(K[t] == hh.k)[0]
out = A*K**hh.theta
cons = out - g[np.int64(Aind),np.int64(Kind)] + (1-hh.delta)*K
inv = out - cons


# Plot the simulation of the economy
t = range(T)
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))

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
plt.savefig("simulation.png")


print("\nThe stochastic steady state is %F, with the true being %F" 
      % (np.mean(K), hh.k_ss))
print("\nThe volatility of output, consumption and investment are %F, %F, and %F." 
      % (np.std(out)*100/np.mean(out),np.std(cons)*100/np.mean(cons), 
         np.std(inv)*100/np.mean(inv)))

