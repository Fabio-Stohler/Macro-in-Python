"""

This solves the Aiyagari model with value function iteration.
It uses Monte-Carlo simulation to aggregate the economy.

"""


import numpy as np
import time
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


class HH:
    
    def __init__(self, theta = 0.36, delta = 0.08, sigma = 3, 
                 beta = 0.96, nz = 7, rho = 0.9, stdev = 0.2,
                 m = 3, nk = 500, kmin = 10**(-5), kmax = 50,
                 T = 1000000):
        
        # Setup parameters
        self.theta, self.delta, self.sigma = theta, delta, sigma
        self.beta, self.nz, self.nk, self.m = beta, nz, nk, m
        self.rho, self.stdev, self.T = rho, stdev, T
        self.stdz = stdev*(1-rho**2)**(1/2)
        self.kmin, self.kmax = kmin, kmax
        
        # Setting up the grid
        self.k = np.zeros(nk)
        for i in range(nk):
            self.k[i] = kmin + kmax/((nk+1)**2.35)*(i**2.35)
        
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
        self.mc = qe.markov.approximation.tauchen(self.rho,self.stdz,
                                             0,self.m,self.nz)
        
        self.sim = self.mc.simulate_indices(self.T, init = int((self.nz-1)/2))
        
        self.P = self.mc.P
        self.l = np.exp(self.mc.state_values)
        inv_l = np.linalg.matrix_power(self.P,1000)
        inv_dist = inv_l[0,:]
        self.l_s = np.dot(self.l, inv_dist)
        return self.P, self.l_s, self.sim


# Generating a class
nz = 7
nk = 500
sigma = 3
rho = 0.6
hh = HH(nz = nz, nk = nk, rho = rho, sigma = sigma)
P, l_s, sim = hh.markov()


# Current level
r = (3.87-1)/100
k_t = hh.interest_reverse(r)


# Value function iteration
@jit 
def Value(r, HH, tolv = 10**(-8), maxiter = 1000):
    w = HH.r_to_w(r)
    sigma, beta, P = HH.sigma, HH.beta, HH.P
    l, k = hh.l, hh.k
    diff    = 10
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


# Simulate the economy and find capital supply
@jit
def simulate(HH, indk, N = 5000):
    mc, l, k, T = HH.mc, HH.sim, HH.k, HH.T
    
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


# Function to solve for the equilibrium
@jit
def Aiyagari(HH, k_t):
    beta = HH.beta
    iter    = 0
    error   = 10
    tol     = 0.01
    
    while error > tol:
        r = HH.interest(k_t)
        if r > 1/beta - 1:
            r = 1/beta - 1
        iter = iter+1
        
        # Value function iteration
        start1 = time.time()
        V, g, indk = Value(r, HH)   
        stop1 = time.time()                   
        
        # Find capital supply
        start2 = time.time()
        a = simulate(HH, indk)
        stop2 = time.time()
        k_s = np.mean(a)
        r1 = HH.interest(k_s)
        error = np.abs(r-r1)*100
        print("\n--------------------------------------------------------------------------------------")
        print("The error in iteration %.0F is %F." % (iter, error))
        print("The capital supply is %F, and the interest rate is %F." %(k_s, r*100))
        print("Value function and simulation took %.3F and %.3F seconds respectively" % ((stop1-start1), (stop2-start2)))
        if error > 2:
            k_t = 0.95*k_t + 0.05*k_s
        elif error > 0.5:
            k_t = 0.99*k_t + 0.01*k_s
        elif error > 0.05:
            k_t = 0.995*k_t + 0.005*k_s
        else:
            k_t = 0.999*k_t + 0.001*k_s
    print("\nThe equilibrium interest rate is %F." % (r*100))
    # Das Ziel sollte 3.87 sein 
    # Resultat war 3.6498 (Check against QE results)
    return V, g, indk, a, r


# Running the function
start = time.time()
V, g, indk, a, r = Aiyagari(hh,k_t)
stop = time.time()
print("Solving the model took %F minutes." %((stop - start)/60))


# Plot the value function and the policy function
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(hh.k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k,g.transpose())
axes[1].plot(hh.k,hh.k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Generate a new distribution
dist = simulate(HH,indk)


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


print("\nThe equilibrium interest rate is %F." % (r*100))
print("Solving the model took %F minutes." %((stop - start)/60))
print("The gini coefficient for the distribution is %F." %(gini(dist)))