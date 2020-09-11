"""

This solves the RBC model with endogeneous labor supply
with the howard improvement algorithm

"""

import time
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import scipy.optimize as opt
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


#create value and policy function objects
g    = np.ones((nz,nk))*c_ss
h    = np.ones((nz,nk))*l_ss
newg = np.copy(g)
newh = np.copy(h)
V = np.copy(g)
newV = np.copy(V)
indk = np.copy(V)


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


# Function to search nearest value on the grid
@jit
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Finding the indicator for the value function iteration
for ik in range(nk):
    for iz in range(nz):
        nearestk = find_nearest(k, zs[iz]*k[ik]**theta*h[iz,ik]**(1-theta) + (1-delta)*k[ik] - g[iz,ik])
        indk[iz,ik] = np.where(k == nearestk)[0]


#tolerance levels
tolv    = 10**-8
maxiter = 1000


# Howard improvement algorithm
@jit
def policy(g = g, h = h, newg = newg, newh = newh, V = V, newV = newV, indk = indk, sigma = sigma, vega = vega, beta = beta, nk = nk, nz = nz, k = k, zs = zs, P = P):
    iter    = 0
    iter1   = iter
    diffg   = 10
    start   = time.time()
    
    while (diffg > tolv and iter<maxiter):
        diffV = 10
        iter1 = iter1+1
        # Generate the value function associated with the policy function
        # We iterate on the value function for a limited amount of time
        # This actually represents modified policy function iteration
        V = np.ones((nz,nk))*c_ss
        iter1 = 0
        maxiter1 = 500
        while (diffV > tolv and iter1<maxiter1):
            iter1 = iter1+1
            for ik in range(nk):
                for iz in range(nz):
                    newV[iz,ik] = (g[iz,ik]**vega*(1-h[iz,ik])**(1-vega))**(1-sigma)/(1-sigma) + beta*(np.dot(P[iz,:], V[:,int(indk[iz,ik])]))
            diffV = np.linalg.norm(newV - V)
            V = np.copy(newV)
            
        
        iter = iter+1
        for iz in range(nz):
            for ik in range(nk):
                l = labor[iz,ik,:]
                c = zs[iz]*k[ik]**theta*l**(1-theta) + (1-delta)*k[ik] - k
                u = (c**vega*(1-l)**(1-vega))**(1-sigma)/(1-sigma)
                u[c<0] = -1000000
                u[l>1] = -1000000
                v = u + beta*(np.dot(P[iz,:], V))
                ind = np.argmax(v)
                indk[iz,ik] = ind
                newg[iz,ik] = c[ind]
                newh[iz,ik] = l[ind]
        diffg = max(np.linalg.norm(newg-g),np.linalg.norm(newh-h))
        g = np.copy(newg)
        h = np.copy(newh)
        print(iter, diffg)
    stop = time.time()                     
    print("\nPolicy function iteration converged after %.0F iterations and %.5F seconds" % (iter, (stop-start)))
    return g, h, V


# Running the function
g, h, V = policy()
print("\nThe population of the matrix took %F seconds." % (stops-starts))


# Transforming the policy function to be for capital
kbar, zbar = np.meshgrid(k,zs)
g = zbar*kbar**theta*h**(1-theta) + (1-delta)*kbar - g


# Plotting the Value function, and the policy function
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
    Kind[t] = np.where(find_nearest(k,K[t]) == k)[0]


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