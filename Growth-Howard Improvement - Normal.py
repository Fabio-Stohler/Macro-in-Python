"""

This solves the stochastic growth model with policy function iteration 

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


class HH:
    
    def __init__(self, theta = 0.4, delta = 0.019, sigma = 2, 
                 beta = 0.99, nz = 21, rho = 0.95, stdev = 0.007,
                 m = 3, nk = 500):
        
        # Setup parameters
        self.theta, self.delta, self.sigma = theta, delta, sigma
        self.beta, self.nz, self.nk, self.m = beta, nz, nk, m
        self.rho, self.stdev = rho, stdev
        
        # Setting up the grid
        self.k_ss = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
        self.y_ss = self.k_ss**theta
        self.c_ss = self.y_ss - delta*self.k_ss
        kmin = 0.8*self.k_ss
        kmax = 1.2*self.k_ss
        self.k = np.linspace(kmin,kmax,nk)
        
        
    def utility(self, c):
        if self.sigma == 1:
            u = np.log(c)
        else:
            u = c**(1-self.sigma)/(1-self.sigma)
        return u
    
    def interest(self,k):
        return self.theta*(k/self.l_s)**(self.theta-1) - self.delta
    
    def interest_reverse(self,r):
        return (self.theta/(r+self.delta))**(1/(1-self.theta))*self.l_s
    
    def r_to_w(self, r):
        return (1-self.theta)*((self.theta/(r+self.delta)))**(self.theta/(1-self.theta))
    
    def markov(self):
        self.mc = qe.markov.approximation.tauchen(self.rho,self.stdev,
                                             0,self.m,self.nz)
        self.P = self.mc.P
        self.zs = np.exp(self.mc.state_values)
        return self.P, self.zs


# Generate a household entity
nk = 1000
hh = HH(nk = nk)
hh.markov()


# Function to search nearest value on the grid
@jit
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Finding the indicator for the value function iteration
def nearest(indk, g, HH):
    theta, delta = HH.theta, HH.delta
    nk, nz = HH.nk, HH.nz
    k, zs = HH.k, HH.zs
    for ik in range(nk):
        for iz in range(nz):
            nearestk = find_nearest(k, zs[iz]*k[ik]**theta + (1-delta)*k[ik] - g[iz,ik])
            indk[iz,ik] = np.where(k == nearestk)[0]
    return indk


# Howard improvement algorithm
@jit
def policy(HH, tolv = 10**(-8), maxiter = 1000):
    
    g    = np.ones((HH.nz,HH.nk))*HH.c_ss
    newg = np.copy(g)
    V = np.copy(g)
    newV = np.copy(V)
    indk = np.copy(V)
    
    indk = nearest(indk, g, HH)
    
    sigma, beta, theta, delta = HH.sigma, HH.beta, HH.theta, HH.delta
    nk, nz = HH.nk, HH.nz
    k, zs, P = HH.k, HH.zs, HH.P
    
    iter    = 0
    iter1   = iter
    diffg   = 10
    start   = time.time()
    
    test1 = (diffg > tolv)
    test2 = (iter<maxiter)
    while (test1 and test2):
        diffV = 10
        iter1 = iter1+1
        # Generate the value function associated with the policy function
        # We iterate on the value function for a limited amount of time
        # This actually represents modified policy function iteration
        V = np.ones((nz,nk))*HH.c_ss
        iter1 = 0
        maxiter1 = 500
        test3 = (diffV > tolv)
        test4 = (iter1 < maxiter1)
        while (test3 and test4):
            iter1 = iter1+1
            for ik in range(nk):
                for iz in range(nz):
                    newV[iz,ik] = g[iz,ik]**(1-sigma)/(1-sigma) + beta*(np.dot(P[iz,:], V[:,int(indk[iz,ik])]))
            diffV = np.linalg.norm(newV - V)
            V = np.copy(newV)
            test3 = (diffV > tolv)
            test4 = (iter1 < maxiter1)
        
        iter = iter+1
        for iz in range(nz):
            for ik in range(nk):
                c = zs[iz]*k[ik]**theta + (1-delta)*k[ik] - k
                u = c**(1-sigma)/(1-sigma)
                u[c<0] = -1000000
                v = u + beta*(np.dot(P[iz,:], V))
                ind = np.argmax(v)
                indk[iz,ik] = ind
                newg[iz,ik] = c[ind]
        diffg = np.linalg.norm(newg-g)
        g = np.copy(newg)
        print(iter, diffg)
        test1 = (diffg > tolv)
        test2 = (iter<maxiter)
    stop = time.time()                     
    print("\nPolicy function iteration converged after %.0F iterations and %.5F seconds" % (iter, (stop-start)))
    return g, V


# Running the function
g, V = policy(hh)


# Transforming the policy function to be for capital
kbar, zbar = np.meshgrid(hh.k,hh.zs)
g = zbar*kbar**hh.theta + (1-hh.delta)*kbar - g


# Plotting the Value function, and the policy function
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,g.transpose())
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Simulate the economy
T = 10000


# Setup the arrays
A = hh.mc.simulate(T, init = hh.mc.state_values[int((hh.nz-1)/2)])
Aind = hh.mc.get_index(A)
A = np.exp(A)
K = np.zeros(T)
Kind = np.copy(K)
Kind[0] = hh.nk/2
K[0] = hh.k[int(Kind[0])]


# Simulating the economy period by period
for t in range(1,T):
    K[t] = g[int(Aind[t-1]),int(Kind[t-1])]
    Kind[t] = np.where(K[t] == hh.k)[0]


# Generate output, consumption and investment
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


# Print the stochastic properties of the economy
print("\nThe stochastic steady state is %F, with the true being %F" % (np.mean(K), hh.k_ss))
print("\nThe volatility of output, consumption and investment are %F, %F, and %F." % (np.std(out)*100/np.mean(out),np.std(cons)*100/np.mean(cons), np.std(inv)*100/np.mean(inv)))