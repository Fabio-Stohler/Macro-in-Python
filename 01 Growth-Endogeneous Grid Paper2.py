# -*- coding: utf-8 -*-
"""

This solves the stochastic growth model with value function iteration

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit
from scipy.interpolate import interp2d, interp1d, Rbf
from scipy.optimize import fsolve


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
        self.kmin = 0.75*self.k_ss
        self.kmax = 1.25*self.k_ss
        self.k = np.linspace(self.kmin,self.kmax,self.nk)
        
    def utility(self, c):
        """Utility function, dependent on consumption c and sigma"""
        if self.sigma == 1:
            util = np.log(c)
        else:
            util = c**(1-self.sigma)/(1-self.sigma)
        return util
    
    def cash_on_hand(self, k, A):
        return (1-self.delta)*k + A*k**self.theta
    
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
hh = HH(nz = 21, nk=50)
mc, P, zs = hh.markov()


# Defining a numerical derivative function for a function f(x+h) - f(x-h) / 2h
def numerical(V, k, HH):
    derivative = np.zeros((HH.nz, HH.nk))
    for i in range(nz):
        for j in range(nk):
            if j == 0:
                derivative[i,j] = (V[i,j+1] - V[i,j])/(k[i,j+1] - k[i,j])
            elif j == (nk-1):
                derivative[i,j] = (V[i,j] - V[i,j-1])/(k[i,j] - k[i,j-1])
            else:
                derivative[i,j] = 0.5*((V[i,j+1] - V[i,j])/(k[i,j+1] - k[i,j])
                                       + (V[i,j] - V[i,j-1])/(k[i,j] - k[i,j-1]))
    return derivative


# Generating the grids
tol = 10**(-6)
maxiter = 100
iter, diff = 0, 10
# Extracting the necessary preference parameters
kmax, kmin = hh.kmax, hh.kmin
nz, nk, Gk = hh.nz, hh.nk, hh.k
theta, beta, delta, sigma = hh.theta, hh.beta, hh.delta, hh.sigma
# Meshgrid
cap, states = np.meshgrid(hh.k, hh.zs)
X = hh.cash_on_hand(cap, states)
Xend = np.copy(X)

# Initial guess for the value function
Vtilde = (states*cap**theta - delta*cap)**(1-sigma)/(1-sigma)
Vtilde_new = np.copy(Vtilde)
Vtilde_prime = np.copy(Vtilde)
Expected = np.copy(Vtilde)
V = np.copy(Vtilde)

test1 = (diff > tol)
test2 = (iter < maxiter)
maxiter = 5
while (test1 and test2):
    iter += 1
    # Numerical derivate of the policy function and optimal consumption
    Vtilde_prime = numerical(Vtilde, Xend, hh)
    for i in range(nz):
        for j in range(nk):
            Expected[i,j] = beta*np.dot(P[i,:], Vtilde_prime[:,j])
    csigma = Expected**(-1/sigma)
    # Defining cash-on-hand
    Xend = csigma + cap
    # Optimal value function (defined on cash on hand)
    Vend = hh.utility(csigma) + beta*Vtilde
    # Interpolating on the exogeneous grid
    for i in range(nz):
        V_inter = interp1d(X[i,:], Vend[i,:])
        V[i,:] = V_inter(X[i,:])
    
    # Generating the expected value
    for i in range(nz):
        for j in range(nk):
            Vtilde_new[i,j] = np.dot(P[i,:], V[:,j])
    
    # Check the convergence criteria
    diff = np.linalg.norm(Vtilde_new - Vtilde)
    Vtilde = np.copy(Vtilde_new)
    test1 = (diff > tol)
    test2 = (iter < maxiter)
    print(iter, diff)


sigma = states*cap**theta + (1-delta)*cap - csigma

               
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,np.transpose(sigma))
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Simulate the economy
T = 5000


# Setup the arrays
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


# Plot the development of the economy
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

