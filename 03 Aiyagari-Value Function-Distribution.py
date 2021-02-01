# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the Aiyagari model with value function iteration.
It uses the invariante simulation to aggregate the economy.

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit

# Supress warning
import warnings
warnings.filterwarnings("ignore")


class HH:
    """
    Setups a class containing all necessary information to solve the
    Aiyagari (1994) model.
    """
    
    def __init__(self, theta = 0.36, delta = 0.08, sigma = 3, 
                 beta = 0.96, nz = 7, rho = 0.9, stdev = 0.2,
                 m = 3, nk = 500, kmin = 10**(-5), kmax = 50):
        """Initialize the class with standard parameters"""
        # Setup parameters
        self.theta, self.delta, self.sigma = theta, delta, sigma
        self.beta, self.nz, self.nk, self.m = beta, nz, nk, m
        self.rho, self.stdev = rho, stdev
        self.stdz = stdev*(1-rho**2)**(1/2)
        self.kmin, self.kmax = kmin, kmax
        
        # Setting up the grid
        self.k = np.zeros(nk)
        for i in range(nk):
            self.k[i] = kmin + kmax/((nk+1)**2.35)*(i**2.35)
        
    def utility(self, c):
        """Utility function depending on the value of sigma"""
        if self.sigma == 1:
            u = np.log(c)
        else:
            u = c**(1-self.sigma)/(1-self.sigma)
        return u
    
    def interest(self,k):
        """Gives back the interest rate, given a capital supply"""
        return self.theta*(k/self.l_s)**(self.theta-1) - self.delta
    
    def interest_reverse(self,r):
        """Given an interest rate, gives back the capital demand"""
        return (self.theta/(r+self.delta))**(1/(1-self.theta))*self.l_s
    
    def r_to_w(self, r):
        """Given an interest rate, the function calculates the wage"""
        return (1-self.theta)*((self.theta/(r+self.delta)))**(self.theta/(1-self.theta))
    
    def markov(self):
        """Approximates the transistion probability of an AR(1) process
           using the methodology of Tauchen (1986) using the quantecon package

           Uses the states, and the transition matrix to give back the
           transition matrix P, as well as invariante labor supply l_s
        """
        self.mc = qe.markov.approximation.tauchen(self.rho,self.stdz,
                                             0,self.m,self.nz)
        self.P = self.mc.P
        self.l = np.exp(self.mc.state_values)
        inv_l = np.linalg.matrix_power(self.P,1000)
        inv_dist = inv_l[0,:]
        inv_l = inv_l / inv_l.sum()
        self.l_s = np.dot(self.l, inv_dist)
        return self.P, self.l_s
    

# Generating a class
nz = 7
nk = 500
sigma = 3
rho = 0.6
hh = HH(nz = nz, nk = nk, rho = rho, sigma = sigma)


# Current level of initial guess
P, l_s = hh.markov()
r = (3.87-1)/100
k_t = hh.interest_reverse(r)


#Value function iteration
@jit 
def Value(r, HH):
    """Given a guess for the interest rate, and a HH class the function
       calculates a optimal policy function, as well as the associated
       value function and indicator function."""
    sigma, beta, P = HH.sigma, HH.beta, HH.P
    l, k = HH.l, HH.k
    w = hh.r_to_w(r)
    
    diff    = 10
    maxiter = 1000
    tolv    = 10**-9
    iter    = 1
    
    # Empty matrices
    g    = np.zeros((nz,nk))
    V    = np.copy(g)
    newV = np.copy(g)
    indk = np.copy(V)
    while (diff > tolv and iter<maxiter):
        for iz in range(nz):
            for ik in range(nk):
                c = w*l[iz] + (1+r)*k[ik] - k
                if sigma != 1:
                    u = c**(1-sigma)/(1-sigma)
                else:
                    u = np.log(np.abs(c))
                u[c<0] = -1000000
                v = u + beta*(np.dot(P[iz,:], V))
                ind = np.argmax(v)
                newV[iz,ik] = v[ind]
                indk[iz,ik] = ind
                g[iz,ik] = k[ind]
        diff = np.linalg.norm(newV-V)
        V = np.copy(newV)
        iter += 1
        #print(iter, diff)
    return V, g, indk


# Calculating the invariate distribution
@jit
def distribution(indk, HH, tol = 10**(-11), maxiter = 50000):
    """Given an indicator function indk, and an instance of a household HH,
       the function calculates an invariante distribution of households over
       the asset and productivity space.
    """
    P, nz, nk = HH.P, HH.nz, HH.nk
    dist = np.ones((nz,nk))/(nz*nk)
    
    error = 1
    iter = 0
    test1 = (error > tol)
    test2 = (iter < maxiter)
    while (test1 and test2):
        distnew = np.zeros((nz,nk))
        for j in range(nk):
            for i in range(nz):
                distnew[:,int(indk[i,j])] = distnew[:,int(indk[i,j])] + dist[i,j]*P[i,:]
        error = np.linalg.norm(distnew - dist)
        dist = np.copy(distnew)
        test1 = (error > tol)
        test2 = (iter < maxiter)
        iter = iter+1
    return dist


# Function to solve for the equilibrium
@jit
def Aiyagari(k_t,HH):
    """Function that completely solves the Aiyagari (1994) model"""
    beta, k = HH.beta, HH.k
    
    iter    = 0
    error   = 10
    tol     = 0.01
    test    = (error > tol)
    while test:
        r = HH.interest(k_t)
        if r > 1/beta - 1:
            r = 1/beta - 1
        iter = iter+1
        
        # Value function iteration
        start1 = time.time()
        V, g, indk = Value(r,HH) 
        stop1 = time.time()                   
        
        # Find capital supply
        start2 = time.time()
        dist = distribution(indk,HH)
        stop2 = time.time()
        k_s = np.sum(k*np.sum(dist, axis = 0))
        r1 = HH.interest(k_s)
        error = np.abs(r-r1)*100
        print("\n--------------------------------------------------------------------------------------")
        print("The error in iteration %.0F is %F." % (iter, error))
        print("The capital supply is %F, and the interest rate is %F." %(k_s, r*100))
        print("Value function and simulation took %.3F and %.3F seconds respectively" % ((stop1-start1), (stop2-start2)))
        if error > 10:
            k_t = 0.9*k_t + 0.1*k_s
        elif error > 5:
            k_t = 0.95*k_t + 0.05*k_s
        elif error > 1:
            k_t = 0.99*k_t + 0.01*k_s
        else:
            k_t = 0.995*k_t + 0.005*k_s
        #elif error > 0.5:
        #    k_t = 0.995*k_t + 0.005*k_s
        #else:
        #    k_t = 0.9975*k_t + 0.0025*k_s
        #k_t = 0.99*k_t + 0.01*k_s
        test = (error > tol)
    print("\nThe equilibrium interest rate is %F." % (r*100))
    # Das Ziel sollte 3.87 sein 
    # Resultat war 3.6498 (Check against QE results)
    return V, g, indk, dist, r


# Running the function
start = time.time()
V, g, indk, dist, r = Aiyagari(k_t,hh)
stop = time.time()
print("Solving the model took %F seconds." %((stop - start)))


# Plot the value function and the policy function
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,g.transpose())
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
#plt.show()
#plt.savefig("convergence.png")


# Generating the distribution
dist1 = distribution(indk, hh)
dist1 = np.sum(dist1, axis = 0)


# Density function
plt.figure(figsize = (15,10))
plt.plot(hh.k, dist1)
plt.xlabel('Asset Value')
plt.ylabel('Frequency')
plt.title('Asset Distribution')
plt.show()


# Monte-Carlo simulation for asset distribution and gini
T = 1000000
mc = hh.mc
k = hh.k
sim = hh.mc.simulate_indices(T, init = int((nz-1)/2))
@jit
def simulate(mc, indk, l = sim, k = k, N = 5000, T = T):
    nz  = np.shape(mc.P)[0]
    T   = np.shape(l)[0]
    m   = T/N
    ind = np.zeros(N)
    for n in range(N):
        l  = np.concatenate((l[int(n*m):T],l[0:int(n*m)]))
        temp = indk[int((nz-1)/2),int(nk/2)]
        for t in range(T):
            temp   = indk[int(l[t]),int(temp)]
        ind[n] = temp
    a = k[np.int64(ind)]
    return a


# Generate a new distribution
dist2 = simulate(mc,indk,N = 10000)


# Plot the distribution
plt.figure(figsize = (15,10))
n, bins, patches = plt.hist(x=dist2, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85,histtype="stepfilled")
plt.xlabel('Asset Value')
plt.ylabel('Frequency')
plt.title('Asset Distribution')
plt.show()
#plt.savefig("distribution.png")


# Function for the gini coefficient
def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


# Print the output (2.28)
print("\nThe equilibrium interest rate is %F." % (r*100))
print("Solving the model took %F minutes." %((stop - start)/60))
print("The gini coefficient for the distribution is %F." %(gini(dist2)))