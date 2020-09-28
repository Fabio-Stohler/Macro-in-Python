"""

This solves the Aiyagari model with the endogeneous grid algorithm
Furthermore, it aggreagtes the economy with the invariate distribution

"""


import numpy as np
import time
import matplotlib.pyplot as plt
import quantecon as qe
import scipy.interpolate as interpl
from numba import jit

# Supress warning
import warnings
warnings.filterwarnings("ignore")


class HH:
    """
    Setups a class containing all necessary information to solve the
    Aiyagari (1994) model.
    """

    def __init__(self, theta=0.36, delta=0.08, sigma=3,
                 beta=0.96, nz=7, rho=0.9, stdev=0.2,
                 m=3, nk=500, kmin=10**(-5), kmax=50):
        """Initializes the class with standard parameters"""
        self.theta, self.delta, self.sigma = theta, delta, sigma
        self.beta, self.nz, self.nk, self.m = beta, nz, nk, m
        self.rho, self.stdev = rho, stdev
        self.stdz = stdev * (1 - rho**2)**(1 / 2)
        self.kmin, self.kmax = kmin, kmax

        # Setting up the grid
        self.k = np.zeros(nk)
        for i in range(nk):
            self.k[i] = kmin + kmax / ((nk + 1)**2.35) * (i**2.35)

    def utility(self, c):
        """Utility function, dependent on the value of sigma"""
        if self.sigma == 1:
            u = np.log(c)
        else:
            u = c**(1 - self.sigma) / (1 - self.sigma)
        return u

    def interest(self, k):
        """Gives back the interest rate given a capital value"""
        return self.theta * (k / self.l_s)**(self.theta - 1) - self.delta

    def interest_reverse(self, r):
        """Gives back the capital value for an interest rate"""
        return (self.theta / (r + self.delta)
                )**(1 / (1 - self.theta)) * self.l_s

    def r_to_w(self, r):
        """Transforms an interest rate into a wage rate"""
        return (1 - self.theta) * ((self.theta / (r + self.delta))
                                   )**(self.theta / (1 - self.theta))

    def markov(self):
        """Approximates the transistion probability of an AR(1) process
           using the methodology of Tauchen (1986) using the quantecon package

           Uses the states, and the transition matrix to give back the
           transition matrix P, as well as invariante labor supply l_s
        """
        self.mc = qe.markov.approximation.tauchen(self.rho, self.stdz,
                                                  0, self.m, self.nz)
        self.P = self.mc.P
        self.labor_states = np.exp(self.mc.state_values)
        inv_l = np.linalg.matrix_power(self.P, 1000)
        inv_dist = inv_l[0, :]
        self.l_s = np.dot(self.labor_states, inv_dist)
        return self.P, self.l_s


# Generating a class
nz = 7
nk = 50
sigma = 3
rho = 0.6
hh = HH(nz=nz, nk=nk, rho=rho, sigma=sigma)


# Current level
P, l_s = hh.markov()
r = (3.87 - 1) / 100
k_t = hh.interest_reverse(r)


# Endogeneous grid algorithm
@jit
def EGM(HH, r, tol = 10**(-6), maxiter = 1000):
    beta = HH.beta
    capital, states = np.meshgrid(HH.k,HH.labor_states)
    g = capital
    a_new = np.copy(g)
    gnew = np.copy(g)
    #g_inter = interpl.Rbf(capital, states, g)
    g_inter = np.ones((nz,nk))
    for i in range(nz):
        g_inter = interpl.interp1d(capital[i,:], g[i,:])

    # Setting up conditions
    error = 10
    iter = 0
    test1 = (error > tol)
    test2 = (iter < maxiter)
    start1 = time.time()
    while test1 and test2:
        iter += 1
        for i in range(nz):
            for j in range(nk):
                gnew[i,j] = (beta*(1+r)*np.dot(P[i,:], g_inter(capital[:,j])**(-sigma)))**(-1/sigma)
                a_new[i,j] = 1/(1+r)*(gnew[i,j] + capital[i,j] - hh.r_to_w(r)*states[i,j])
                # We have incomplete markets, meaning we have a borrowing limit
                #if a_new[i,0] > 0:
                    #    a_new[i,0] = hh.kmin
                    #    gnew[i,0] = hh.r_to_w(r)*states[i,0]
                    #    if a_new[i,j] < 0:
                        #        gnew[i,j] = hh.r_to_w(r)*states[i,j] - capital[i,j]
        for i in range(nz):
            g_inter = interpl.interp1d(a_new[i,:], gnew[i,:], bounds_error = False, fill_value = "extrapolate")
            gnew[i,:] = g_inter(capital[i,:])
        #g_inter = interpl.Rbf(a_new, states, gnew, function = "cubic")
        #gnew = g_inter(capital, states)
        # Comparison of the criteria
        error = np.linalg.norm(g - gnew)
        g = np.copy(gnew)
        test1 = (error > tol)
        test2 = (iter < maxiter)
        print(iter, error)
    stop1 = time.time()
    print("\nEGM converged after %F seconds and %.0F iterations." % ((stop1-start1),iter))
    return g, a_new

# We need to transform the policy function into a function on capital
g, a_new = EGM(hh, r)
capital, states = np.meshgrid(hh.k,hh.labor_states)
sigma = g
g = hh.r_to_w(r)*states + (1+r)*a_new - sigma


# Function to setup J given a guessed policy function
@jit
def get_J(g, HH):
    """Using a policy function, as well as a HH class object, the function
       computes the J matrix according to Ljundgqvist and Sargent to
       calculate a new value function by matrix inversion."""
    # Extracting the parameters
    k, nz, nk = HH.k, HH.nz, HH.nk
    J = np.zeros((nz, nk, nk))
    for i in range(nz):
        for j in range(nk):
            J[i, j, :] = (g[i, j] == k)
    return J


# Generate a reward vector
@jit
def reward(r, HH, g):
    """Given an interest rate r, an HH class instance and a policy function g,
       the function gives back a vector of utility required for the get_Value
       function."""
    nk, nz, k, labor_states = HH.nk, HH.nz, HH.k, HH.labor_states
    w = hh.r_to_w(r)
    # Calculate the utility
    re = hh.utility((1 + r) * k.reshape(1, nk) + w
                    * labor_states.reshape(nz, 1) - g)
    # Transform into the required format for the function
    return re.flatten()


# Function to setup a similar matrix as the Q matrix to Ljundgqvist and
# Sargent:
@jit
def get_Q(P, J):
    """Given the transition matrix P and the J matrix from the get_J function
       this function gives back the stochastic transition matrix Q as in
       Ljungqvist and Sargent (2012)."""
    # Setup empty matrices
    shape1 = np.shape(P)[0]
    shape2 = np.shape(J)[1]
    shape3 = shape1 * shape2
    Q = np.zeros((shape3, shape3))
    for i in range(shape1):
        for j in range(shape1):
            pos11 = int(i * shape2)
            pos12 = int((i + 1) * shape2)
            pos21 = int(j * shape2)
            pos22 = int((j + 1) * shape2)
            Q[pos11:pos12, pos21:pos22] = P[i, j] * J[i, :]
    return Q


# Generate a new value function
@jit
def get_value(Q, re, HH):
    """Given the stochastic transition matrix Q, a reward vector re, and
       a HH class instance this function gives back a new value function."""
    nz, nk, beta = HH.nz, HH.nk, HH.beta
    matrix = np.eye(np.shape(Q)[0]) - beta * Q
    inverse = np.linalg.inv(matrix)
    # Calculating new value function as a vector
    v = np.dot(inverse, re)
    # Putting the vector back in required shape
    v = v.reshape(nz, nk)
    return v


# Extrcting a policy function, given a Value function
@jit
def get_policy(V, r, HH):
    """Given a value function V, an interest rate r, as well as a HH class,
       the function computes a new policy function g, as well as the
       corresponding indicator function indk"""
    # Unpacking of parameters
    nz, nk, P = HH.nz, HH.nk, HH.P
    labor, k = HH.labor_states, HH.k
    beta = HH.beta
    R = 1 + r
    w = HH.r_to_w(r)

    # Setting up the empty matrices
    g = np.zeros((nz, nk))
    indk = np.copy(V)
    for iz in range(nz):
        for ik in range(nk):
            # Calculating consumption and its corresponding utility
            c = w * labor[iz] + R * k[ik] - k
            u = HH.utility(c)
            # Penalize negative consumption
            u[c < 0] = -1000000
            # Get value function, maximize and get new policy function
            v = u + beta * (np.dot(P[iz, :], V))
            ind = np.argmax(v)
            indk[iz, ik] = ind
            g[iz, ik] = k[ind]
    return g, indk


# Calculating the invariate distribution
@jit
def distribution(indk, HH, tol = 10**(-11), maxiter = 50000):
    
    P, nk = HH.P, HH.nk
    nz = np.shape(P)[0]
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


# Generating the value function associated with the policy function
J = get_J(g, hh)
re = reward(r, hh, g)
Q = get_Q(hh.P, J)
V = get_value(Q,re,hh)
g2, indk = get_policy(V, r, hh) 







# Plot the value function and the policy function
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
axes[0].plot(hh.k, V.transpose())
axes[0].set_title("Value functions")

axes[1].plot(hh.k, g.transpose())
axes[1].plot(hh.k, hh.k)
axes[1].set_title('Policy functions')
plt.show()
plt.savefig("convergence.png")


# Generating the distribution
dist1 = distribution(indk, hh)
dist1 = np.sum(dist1, axis=0)


# Density function
plt.figure(figsize=(15, 10))
plt.plot(hh.k, dist1)
plt.xlabel('Asset Value')
plt.ylabel('Frequency')
plt.title('Asset Distribution')
plt.show()


# Monte-Carlo simulation for asset distribution and gini
T = 1000000
mc = hh.mc
k = hh.k
sim = hh.mc.simulate_indices(T, init=int((nz - 1) / 2))
@jit
def simulate(mc, indk, labor_sim=sim, k=k, N=10000, T=T):
    """Simulate a cross-section of households over time to derive an
       invariante distribution of households assets for later plotting."""
    nz = np.shape(mc.P)[0]
    m = T / N
    ind = np.zeros(N)
    for n in range(N):
        labor_sim = np.concatenate((labor_sim[int(n * m):T],
                                    labor_sim[0:int(n * m)]))
        temp = indk[int((nz - 1) / 2), int(nk / 2)]
        for t in range(T):
            temp = indk[int(labor_sim[t]), int(temp)]
        ind[n] = temp
    a = k[np.int64(ind)]
    return a


# Generate a new distribution
dist2 = simulate(mc, indk, N=10000)


# Plot the distribution
plt.figure(figsize=(15, 10))
n, bins, patches = plt.hist(x=dist2, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85, histtype="stepfilled")
plt.xlabel('Asset Value')
plt.ylabel('Frequency')
plt.title('Asset Distribution')
plt.show()
plt.savefig("distribution.png")


# Print the output
print("\nThe equilibrium interest rate is %F." % (r * 100))
print("Solving the model took %F minutes." % ((stop - start) / 60))
print("The gini coefficient for the distribution is %F."
      % (qe.gini_coefficient(dist2)))
