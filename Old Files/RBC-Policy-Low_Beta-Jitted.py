# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the RBC model with value function iteration and exogenous labor
supply

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


#parameters
theta = 0.4; 
delta = 0.019;
sigma = 2;
beta  = 0.9;
nz    = np.int(21);
rho   = 0.95;
stdz  = np.sqrt(0.000049);
m     = 3;
sims  = 10;
nk   = np.int(500);

#discretizing the grid
mc = qe.markov.approximation.tauchen(rho,stdz,0,m,nz)
P = mc.P
zs = np.exp(mc.state_values)
#zs = mc.state_values+1      # As alternative symetric shocks

#steady state quantities
ky_ss = (beta*theta)/(1-beta*(1-delta));
y_ss  = ((beta*theta)/(1-beta*(1-delta)))**(theta/(1-theta))
k_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
c_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))*(1-beta*(1-delta)-beta*theta*delta)/(beta*theta)

#discretizing the k grid
kmin = 0.8*k_ss
kmax = 1.2*k_ss
k = np.linspace(kmin,kmax,nk)

#tolerance levels
tolv    = 10**-8
maxiter = 1000

kbar, zbar = np.meshgrid(k,zs)
interest = zbar*theta*kbar**(theta-1) + (1-delta)

#Policy function iteration
@jit
def policy(k = k, zs = zs, interest = interest, theta = theta, delta = delta, sigma = sigma, beta = beta, P = P, nz = nz, nk = nk, c_ss = c_ss):
    g    = np.ones((nz,nk))*c_ss
    newg = np.copy(g)
    iter    = 0
    diffg   = 10
    while (diffg > tolv and iter<maxiter):
        iter = iter+1
        for iz in range(nz):
            for ik in range(nk):
                res = (zs[iz]*k[ik]**theta + (1-delta)*k[ik] - k) - (beta*np.dot(P[iz,:], interest*g**(-sigma)))**(-1/sigma)
                res = res**4
                #error = min(res)
                ind = np.argmin(res)
                newg[iz,ik] = zs[iz]*k[ik]**theta + (1-delta)*k[ik] - k[ind]
        diffg = np.linalg.norm(newg-g)
        g = np.copy(newg)
        print(iter, diffg)
    print("\nPolicy function iteration converged after %.0F iterations." % (iter))
    return g


start = time.time()           
g = policy()
stop = time.time() 
print("\nExecution took %F seconds." % (stop - start))

V = np.zeros((nz,nk))
Vnew = np.copy(V)
diffv = 10
iter = 0
while (diffv > tolv and iter < maxiter):
    for iz in range(nz):
        for ik in range(nk):
            iter = iter + 1
            c = zs[iz]*k[ik]**theta + (1-delta)*k[ik] - k
            v = c**(1-sigma)/(1-sigma) + beta*np.dot(P[iz,:],V)
            v[c<0] = -1000000000
            Vnew[iz,ik] = max(v)
    diffv = np.linalg.norm(Vnew-V)
    print(diffv)
    V = np.copy(Vnew)
    
        
g = zbar*kbar**theta + (1-delta)*kbar - g
       

#print(toc-tic)                 
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(k,g.transpose())
axes[1].plot(k,k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")



# Simulate the economy
T = 5000

# Setup the arrays
A = mc.simulate(T, init = mc.state_values[int((nz-1)/2)])
Aind = mc.get_index(A)
A = np.exp(A)
K = np.zeros(T)
Kind = np.copy(K)
Kind[0] = nk/2
K[0] = k[int(Kind[0])]


# Simulating the economy period by period
for t in range(1,T):
    K[t] = g[int(Aind[t-1]),int(Kind[t-1])]
    Kind[t] = np.where(K[t] == k)[0]


out = A*K**theta
cons = out - g[np.int64(Aind),np.int64(Kind)] + (1-delta)*K
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


print("\nThe stochastic steady state is %F, with the true being %F" % (np.mean(K), k_ss))
print("\nThe volatility of output, consumption and investment are %F, %F, and %F." % (np.std(out)*100/np.mean(out),np.std(cons)*100/np.mean(cons), np.std(inv)*100/np.mean(inv)))

# Extension: Solving it with a lower state space, but with interpolation
# Put labor supply
# Paralize this code
# How to put a borrowing limit
  
