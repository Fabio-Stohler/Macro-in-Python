# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the Aiyagari model with value function iteration.
It uses Monte-Carlo simulation to aggregate the economy.

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


#parameters
theta = 0.36 
delta = 0.08
sigma = 5
beta  = 0.96
nz    = np.int(7)
rho   = 0.6
stdev = 0.2
stdz  = stdev*(1-rho**2)**(1/2)
m     = 3
nk   = np.int(500)


#discretizing the grid
mc = qe.markov.approximation.tauchen(rho,stdz,0,m,nz)
P = mc.P
l = np.exp(mc.state_values)
inv_l = np.linalg.matrix_power(P,1000)
inv_dist = inv_l[0,:]
l_s = np.dot(l, inv_dist)


# Drawing realizations
T = 1000000
sim = mc.simulate_indices(T, init = int((nz-1)/2))


#discretizing the k grid
k_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
kmin = 10**(-5)
kmax = 100
k = np.zeros(nk)
for i in range(nk):
    k[i] = kmin + kmax/((nk+1)**2.35)*(i**2.35)


# Current level
r = (3.87-1)/100
k_t = (theta/(r+delta))**(1/(1-theta))*l_s


#Value function iteration
@jit 
def Value(w, R, sigma = sigma, beta = beta, P = P):
    diff    = 10
    maxiter = 1000
    tolv    = 10**-10
    iter    = 1
    
    # Empty matrices
    g    = np.zeros((nz,nk))
    V    = np.copy(g)
    newV = np.copy(g)
    indk = np.copy(V)
    while (diff > tolv and iter<maxiter):
        for iz in range(nz):
            for ik in range(nk):
                c = w*l[iz] + R*k[ik] - k
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
def distribution(indk, P = P, nz = nz, nk = nk, tol = 10**(-12), maxiter = 50000):
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
def Aiyagari(l, k_t):
    iter    = 0
    error   = 10
    tol     = 0.01
    
    while error > tol:
        r = theta*(k_t/l_s)**(theta-1) - delta
        if r > 1/beta - 1:
            r = 1/beta - 1
        R = 1+r
        w = (1-theta)*((theta/(r+delta)))**(theta/(1-theta))
        iter = iter+1
        
        # Value function iteration
        start1 = time.time()
        V, g, indk = Value(w,R)   
        stop1 = time.time()                   
        
        # Find capital supply
        start2 = time.time()
        dist = distribution(indk)
        stop2 = time.time()
        k_s = np.sum(k*np.sum(dist, axis = 0))
        r1 = theta*(k_s/l_s)**(theta-1) - delta
        error = np.abs(r-r1)*100
        print("\n--------------------------------------------------------------------------------------")
        print("The error in iteration %.0F is %F." % (iter, error))
        print("The capital supply is %F, and the interest rate is %F." %(k_s, r*100))
        print("Value function and simulation took %.3F and %.3F seconds respectively" % ((stop1-start1), (stop2-start2)))
        k_t = 0.99*k_t + 0.01*k_s
    print("\nThe equilibrium interest rate is %F." % (r*100))
    # Das Ziel sollte 3.87 sein 
    # Resultat war 3.6498 (Check against QE results)
    return V, g, indk, dist, r


# Running the function
start = time.time()
V, g, indk, dist, r = Aiyagari(l,k_t)
stop = time.time()
print("Solving the model took %F seconds." %((stop - start)))


# Plot the value function and the policy function
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(k,g.transpose())
axes[1].plot(k,k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Generate a new distribution
dist = np.sum(dist, axis = 0)
dist = dist*k


# Plot the distribution
plt.figure(figsize = (15,10))
n, bins, patches = plt.hist(x=dist, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85,histtype="stepfilled")
plt.xlabel('Asset Value')
plt.ylabel('Frequency')
plt.title('Asset Distribution')
plt.show()
plt.savefig("distribution.png")


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


# Resultat sollte 0.32 sein für gamma = 5, rho = 0.6, und sigma = 0.2
print("The gini coefficient for the distribution is %F." %(gini(dist)))