# -*- coding: utf-8 -*-
"""

This solves the stochastic growth model with value function iteration

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import fsolve
from scipy.interpolate import interp1d


class HH():
    
    def __init__(self,theta = 0.4, beta = 0.99, delta = 0.019, sigma = 1, nk = 500):
        """Initialize the class with standard parameters"""
        self.theta, self.beta, self.delta = theta, beta, delta
        self.sigma, self.nk =  sigma, nk
        
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
    
    def utility_prime(self, c):
        """Derivative of the utility function"""
        return c**(-self.sigma)
    
    def utility_prime_inverse(self, marginal):
        """Given marginal utility, gives back consumption"""
        return marginal**(-1/self.sigma)


# Setting up stuff
nk = 500
hh = HH(nk = nk)
nk, beta, sigma = hh.nk, hh.beta, hh.sigma
k = hh.k
# Cash in hand
Gy = k**hh.theta + (1-hh.delta)*k
Gyend = np.copy(Gy)
kend = np.copy(Gy)
interest = hh.theta*k**(hh.theta-1) + (1-hh.delta)
# Initial guess for the policy function is all capital
g = k
g_new = np.copy(g)
# Passing an interpolated function into the loop
g_inter = interp1d(k, g)
tol = 10**(-8)
maxiter = 5000
error = 1
iter = 0
test1 = (maxiter > iter)
test2 = (error > tol)
start1 = time.time()
while test1 and test2:
    iter += 1
    for j in range(nk):
        g_new[j] = hh.utility_prime_inverse(beta*interest[j]*hh.utility_prime(g[j]))
        Gyend[j] = g_new[j] + k[j]
        res = lambda k: Gyend[j] - k**(hh.theta) - (1-hh.delta)*k
        kend[j] = fsolve(res, k[j])
    
    g_inter = interp1d(kend,g_new,bounds_error = False, fill_value="extrapolate")
    error = np.linalg.norm(g_inter(k) - g)
    g = g_inter(k)
    test1 = (maxiter > iter)
    test2 = (error > tol)
    if iter % 50 == 0:
        print(iter, error)
stop1 = time.time()
print("\nEGM converged after %F seconds and %.0F iterations." % ((stop1-start1),iter))


# Given the optimal policy, generate a value function
error = 1
tol = 10**(-5)
test = (error > tol)
V = np.zeros(nk)
V_new = np.copy(V)
while test:
    V_new = hh.utility(g) + beta*V
    error = np.linalg.norm(V_new - V)
    V = V_new
    test = (error > tol)
       

# Policy function for consumption into policy function for capital
sigma = g
g = hh.k**hh.theta + (1-hh.delta)*hh.k - sigma
    

#print(toc-tic)                 
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,g.transpose())
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Setup the arrays
T = 200
g = interp1d(k,g,bounds_error = False, fill_value="extrapolate")
sigma = interp1d(k,sigma,bounds_error = False, fill_value="extrapolate")
K = np.zeros(T)
K[0] = hh.k[int(nk/100)]


# Simulating the economy period by period
for t in range(1,T):
    K[t] = g(K[t-1])
out = K**hh.theta
cons = out - g(K) + (1-hh.delta)*K
cons1 = sigma(K)
inv = out - cons


# Plot the development of the economy
t = range(T)
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))

axes[0].plot(t, K, label = "Development of capital")
axes[0].plot(t, np.ones(T)*hh.k_ss, label = "Steady state of capital")
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


# Calculating the steady state
res = lambda k: g(k) - k
steady_state = fsolve(res, hh.k_ss)
print("\nThe numerical steady state is %F, while the true is %F." %(steady_state, hh.k_ss))


# Calculating the Euler equation error