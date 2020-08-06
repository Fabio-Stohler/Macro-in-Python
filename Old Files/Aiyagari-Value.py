# -*- coding: utf-8 -*-
"""
Spyder Editor

This solves the Aiyagari model with value function iteration

"""

import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe


#parameters
theta = 0.36 
delta = 0.08
sigma = 3
beta  = 0.96
nk    = 100
nz    = np.int(3)
rho   = 0.6
stdev = 0.2
stdz  = stdev*(1-rho**2)**(1/2)
m     = 3
nk   = np.int(50)


#discretizing the grid
mc = qe.markov.approximation.tauchen(rho,stdz,0,m,nz)
P = mc.P
l = np.exp(mc.state_values)
inv_l = np.linalg.matrix_power(P,1000)
inv_dist = inv_l[0,:]
l_s = np.dot(l, inv_dist)


#discretizing the k grid
k_ss  = ((beta*theta)/(1-beta*(1-delta)))**(1/(1-theta))
kmin = 10**(-5)
kmax = 50
k = np.zeros(nk)
for i in range(nk):
    k[i] = kmin + kmax/((nk+1)**2.35)*(i**2.35)


#create value and policy function objects
g    = np.zeros((nz,nk))
newg = np.copy(g)
V    = np.copy(g)
newV = np.copy(g)
indk = np.copy(V)


# Current level
k_t = 6
r = theta*(k_t/l_s)**(theta-1) - delta


#tolerance levels
tolv    = 10**-8
tol     = 0.0001
maxiter = 1000
iter    = 0
diff    = 10
error   = diff


while error > tol:  
    R = 1+r
    w = (1-theta)*((theta/(r+delta)))**(theta/(1-theta))
    iter = iter+1
    
    #Value function iteration
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
        #print(iter, diff)
                    
    
    # Simulate an economy forward
    T = 10000
    N = 500
    l = np.zeros((N,T))
    ind = np.copy(l)
    for n in range(N):
        l[n,:] = mc.simulate(T, init = mc.state_values[int((nz-1)/2)])
        l[n,:] = mc.get_index(l[n,:])
        temp = indk[int((nz-1)/2),int(nk/2)]
        for t in range(T):
            temp   = indk[int(l[n,t]),int(temp)]
            ind[n,t] = temp
    a = k[np.int64(ind)]    
    k_s = np.mean(a[:,T-1])
    k_d = (theta/(delta + r))**(1/(1-theta))*l_s
    r1 = theta*(k_s/l_s)**(theta-1) - delta
    error = np.abs(r-r1)
    print("\nThe error in iteration %.0F is %F." % (iter, error))
    print("\nThe capital supply is %.5F, and the interest rate is %.5F." %(k_s, r*100))
    r = 0.95*r + 0.05*r1
    
print("\nThe equilibrium interest rate is %F." % r)
# Das Ziel sollte 3.87 sein


# Plot the results
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(15,10))
axes[0].plot(k,V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(k,g.transpose())
axes[1].plot(k,k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")