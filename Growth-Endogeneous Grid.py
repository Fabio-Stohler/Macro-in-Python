# -*- coding: utf-8 -*-
"""

This solves the stochastic growth model with value function iteration

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


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
        self.kmin = 0.5*self.k_ss
        self.kmax = 1.5*self.k_ss
        self.k = np.linspace(self.kmin,self.kmax,self.nk)
        
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
hh = HH(nz = 21, nk=250)
mc, P, zs = hh.markov()


import scipy.interpolate as interp
def numerical(value, capital, h = 0.01):
    size = np.shape(value)[0]
    der  = np.copy(value)
    y_app = interp.interp1d(capital, value)
    for i in range(size):
        if i == 0:
            der[i] = ((y_app(capital[i+1])-y_app(capital[i]))/(capital[i+1]-capital[i]))
        elif i == (size-1):
            der[i] = ((y_app(capital[i])-y_app(capital[i-1]-h))/(capital[i]-capital[i-1]))
        else:
            der[i] = ((y_app(capital[i+1])-2*y_app(capital[i])+y_app(capital[i-1]))/(capital[i+1]-2*capital[i]+capital[i-1]))
    return der



# Defining a numerical derivative function for an array
# If convergence in the algorithm does not occure, we need to update here
# Actual formular for derivative: f(x+h) - f(x-h) / 2h
def numerical(value, capital):
    size = np.shape(value)[0]
    der  = np.copy(value)
    for i in range(size):
        if i == 0:
            der[i] = ((value[1] - value[0])/(capital[1] - capital[0]))
        elif i == (size-1):
            der[i] = ((value[i] - value[i-1])/(capital[i] - capital[i-1]))
        else:
            der[i] = ((value[i+1] - value[i-1])/(capital[i+1]-capital[i-1]))
    return der




# Generating the grids
tol = 10**(-6)
maxiter = 100
iter, diff = 0, 10
kmax, kmin = hh.kmax, hh.kmin
nz, nk, Gk = hh.nz, hh.nk, hh.k
theta, beta, delta, sigma = hh.theta, hh.beta, hh.delta, hh.sigma
cap, states = np.meshgrid(hh.k, hh.zs)
Gy = states*cap**theta + (1-delta)*cap

# Initial guess for the value function
Vtilde = cap**(0.5)*states
Vtilde_new = np.copy(Vtilde)
Vtilde_prime = np.copy(Vtilde)
V = np.copy(Vtilde)

test1 = (diff > tol)
test2 = (iter < maxiter)
maxiter = 5
while (test1 and test2):
    iter += 1
    # Numerical derivate of the policy function and optimal consumption
    Vtilde_prime = np.zeros((nz,nk))
    Vtilde_prime[:,1:nk] += np.diff(Vtilde)
    Vtilde_prime[:,0:nk-1] += np.diff(Vtilde)
    Vtilde_prime[:,1:nk-1] = Vtilde_prime[:,1:nk-1]/(2*(kmax-kmin)/nk)
    
    cprime = Vtilde_prime**(-1/sigma)
    Yend = cprime + Gk
    Vend = cprime**(1-sigma)/(1-sigma) + Vtilde
    
    for i in range(nz):
        V[i,:] = np.interp(Gy[i,:], Yend[i,:], Vend[i,:])
    
    for i in range(nz):
        for j in range(nk):
            Vtilde_new[i,j] = beta*np.dot(P[i,:], V[:,j])
    
    diff = np.linalg.norm(Vtilde_new - Vtilde)
    Vtilde = Vtilde_new
    test1 = (diff > tol)
    test2 = (iter < maxiter)
    print(iter, diff)


# Running the function

       

#print(toc-tic)                 
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,cprime.transpose())
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

