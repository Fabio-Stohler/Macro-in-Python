# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the RBC model with value function iteration and exogenous 
labor supply

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe


start = time.time()


#parameters
theta = 0.4
delta = 0.019
sigma = 2
beta  = 0.99
nk    = 100
nz    = np.int(9)
rho   = 0.95
stdz  = np.sqrt(0.000049)
m     = 3
nk   = np.int(50)

#discretizing the grid
mc = qe.markov.approximation.tauchen(rho,stdz,0,m,nz)
P = mc.P
zs = np.exp(mc.state_values)


#steady state quantities
k_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
y_ss  = k_ss**theta
c_ss  = k_ss*(1-beta*(1-delta)-beta*theta*delta)/(beta*theta)


#discretizing the k grid
kmin = 0.8*k_ss
kmax = 1.2*k_ss
k = np.linspace(kmin,kmax,nk)


#create value and policy function objects
u    = c_ss**(1-sigma)/(1-sigma)
V    = np.ones((nz,nk))*u/(1-beta)
v    = np.zeros((nz,nk))
g    = np.zeros((nz,nk))
newV = np.zeros((nz,nk))


#tolerance levels
tolv    = 10**-8
maxiter = 5000
iter    = 0
diffV   = 10


#Value function iteration
while (diffV > tolv and iter<maxiter):
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


stop = time.time()        


print("\nValue function iteration converged after %.0F iterations and %.5F seconds" % (iter, (stop-start)))
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


# Transform the policy function to a policy function of consumption
kbar, abar = np.meshgrid(k, zs)
h = abar*kbar**theta - g + (1-delta)*k

# Calculating the euler equation error
from scipy.interpolate import interp2d
pol = interp2d(abar, kbar, h, kind='cubic')

def EER(aval,kval):
    g = interp2d(abar,kbar,g,kind='cubic')
    h = interp2d(abar,kbar,h,kind='cubic')
    kplus = h(aval,kval)
    EV = beta*((1-delta)*np.dot(abar,kplus(abar,kval)**(theta-1)))**(-sigma)*g(abar,kbar)
    