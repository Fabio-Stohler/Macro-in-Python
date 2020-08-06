# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the Aiyagari model with value function iteration

"""

import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


#parameters
theta = 0.36 
delta = 0.08
sigma = 3
beta  = 0.96
nk    = 100
nz    = np.int(7)
rho   = 0.6
stdev = 0.2
stdz  = stdev*(1-rho**2)**(1/2)
m     = 3
nk   = np.int(200)


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
kmax = 50
k = np.zeros(nk)
for i in range(nk):
    k[i] = kmin + kmax/((nk+1)**2.35)*(i**2.35)


# Current level
k_t = 6
r = theta*(k_t/l_s)**(theta-1) - delta
r = 3.87/100

#Value function iteration
@jit 
def Value(w, R, sigma = sigma, beta = beta, P = P):
    diff    = 10
    maxiter = 1000
    tolv    = 10**-8
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


# Simulate the economy and find capital supply
@jit
def simulate(mc, indk, l=sim, k=k):
    nz  = np.shape(mc.P)[0]
    T   = np.shape(l)[0]
    N   = 5000
    m   = T/N
    ind = np.zeros(N)
    for n in range(N):
        l  = np.concatenate((l[int(n*m):T],l[0:int(n*m)]))
        temp = indk[int((nz-1)/2),int(nk/2)]
        for t in range(T):
            temp   = indk[int(l[t]),int(temp)]
        ind[n] = temp
    a = k[np.int64(ind)]
    k_s = np.mean(a)
    return k_s


# Function to solve for the equilibrium
#@jit
def Aiyagari(l, r):
    iter    = 0
    error   = 10
    tol     = 0.01
    
    while error > tol:
        R = 1+r
        w = (1-theta)*((theta/(r+delta)))**(theta/(1-theta))
        iter = iter+1
        
        # Value function iteration
        V, g, indk = Value(w,R)                      
        
        # Find capital supply
        k_s = simulate(mc, indk)
        
        k_d = (theta/(delta + r))**(1/(1-theta))*l_s
        r1 = theta*(k_s/l_s)**(theta-1) - delta
        if r1 > 1/beta-1:
            r1 = 1/beta-1
        error = np.abs(k_s-k_d)
        print("\nThe error in iteration %.0F is %F." % (iter, error))
        print("\nThe capital supply is %.5F, and the interest rate is %.5F." %(k_s, r*100))
        r = 0.95*r + 0.05*r1
    print("\nThe equilibrium interest rate is %F." % r)
    # Das Ziel sollte 3.87 sein
    return V, g, indk


# Running the function
V, g, indk = Aiyagari(l,r)


# Plot the results
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(k,g.transpose())
axes[1].plot(k,k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")