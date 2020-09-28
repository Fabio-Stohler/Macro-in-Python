# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the RBC model with value function iteration and 
endogeneous labor supply

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import quantecon as qe
from numba import jit

# Supress warning
import warnings
warnings.filterwarnings("ignore")


#parameters
theta = 0.4; 
delta = 0.019;
sigma = 2;
vega  = 0.36;
beta  = 0.99;
nk    = 100;
nz    = np.int(21);
rho   = 0.95;
stdz  = np.sqrt(0.000049);
m     = 3;
sims  = 10;
nk   = np.int(250);

#discretizing the grid
mc = qe.markov.approximation.tauchen(rho,stdz,0,m,nz)
P = mc.P
zs = np.exp(mc.state_values)
#zs = mc.state_values+1      # As alternative symetric shocks

#steady state quantities
kl_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
yl_ss  = kl_ss**theta
cl_ss  = yl_ss - delta*kl_ss
ll_ss  = (1-vega)/vega*cl_ss/(1-theta)*kl_ss**(-theta)
l_ss   = 1/(1+ll_ss)
k_ss   = kl_ss*l_ss
y_ss   = k_ss**theta
c_ss   = y_ss - delta*k_ss


#discretizing the k grid
kmin = 0.8*k_ss
kmax = 1.2*k_ss
k = np.linspace(kmin,kmax,nk)


#tolerance levels
tolv    = 10**-8
maxiter = 5000

starts = time.time()
# Setting up the matrixes as a function of future k
consumption = np.zeros((nz,nk,nk))
labor = np.copy(consumption)
for iz in range(nz):
    for ik in range(nk):
        for jk in range(nk):
            res = lambda l: zs[iz]*k[ik]**theta*l**(1-theta) + (1-delta)*k[ik] - k[jk] - vega/(1-vega)*(1-l)*(1-theta)*zs[iz]*(k[ik]/l)**theta
            labor[iz,ik,jk] = opt.fsolve(res, l_ss)
            consumption[iz,ik,jk] = vega/(1-vega)*(1-labor[iz,ik,jk])*(1-theta)*(k[ik]/labor[iz,ik,jk])**(theta)
stops = time.time()

#Value function iteration
@jit
def value_function(nz = nz, nk = nk, consumption = consumption, labor = labor, vega = vega, beta = beta, sigma = sigma, P = P, k = k):    
    iter    = 0
    diffV   = 10
    u    = c_ss**(1-sigma)/(1-sigma)
    V    = np.ones((nz,nk))*u/(1-beta)
    v    = np.zeros((nz,nk))
    g    = np.zeros((nz,nk))
    h    = np.copy(g)
    newV = np.zeros((nz,nk))
    start = time.time()
    while (diffV > tolv and iter<maxiter):
        iter = iter+1
        for iz in range(0,nz):
            for ik in range(0,nk):
                c = consumption[iz,ik,:]
                l = labor[iz,ik,:]
                u = (c**vega*(1-l)**(1-vega))**(1-sigma)/(1-sigma)
                u[c<0] = -1000000000
                u[l>1] = -1000000000
                v = u+beta*(np.dot(P[iz,:],V))
                newV[iz,ik] = max(v)
                ind         = np.argmax(v)
                g[iz,ik]    = k[ind]
                h[iz,ik]    = l[ind]
        diffV = np.linalg.norm(newV-V)
        V = newV.copy()
        print(iter, diffV)
    stop = time.time()
    print("\nValue function iteration converged after %.0F iterations and %.5F seconds" % (iter, (stop-start)))
    return g, h, V


# Running the function
g, h, V = value_function()
print("\nThe population of the matrix took %F seconds." % (stops-starts))

# Plotting the value function           
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


lab = h[np.int64(Aind),np.int64(Kind)]
out = A*K**theta*lab**(1-theta)
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
print("\nThe mean of consumption, investment, capital, and labor in relation to output are %F, %F, %F, and %F." % (np.mean(cons*100/out), np.mean(inv*100/out), np.mean(K*100/(4*out)), np.mean(lab*100)))
print("\nThe CV of consumption, investment and labor in relation to the CV of output are %F, %F, and %F." % ((np.std(cons)*100/np.mean(cons))/(np.std(out)*100/np.mean(out)),(np.std(inv)*100/np.mean(inv))/(np.std(out)*100/np.mean(out)),(np.std(lab)*100/np.mean(lab))/(np.std(out)*100/np.mean(out))))
print("\nThe correlation of consumption, investment and labor with output are %F, %F, and %F." %(np.corrcoef(out,cons)[0,1], np.corrcoef(out,inv)[0,1], np.corrcoef(out, lab)[0,1]))