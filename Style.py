"""

This solves the Aiyagari model with policy function iteration
Furthermore, it aggreagtes the economy with the invariate distribution

"""


import numpy as np
import time
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit


class HH:

    def __init__(self, theta=0.36, delta=0.08, sigma=3,
                 beta=0.96, nz=7, rho=0.9, stdev=0.2,
                 m=3, nk=500, kmin=10**(-5), kmax=50):

        # Setup parameters
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
        if self.sigma == 1:
            u = np.log(c)
        else:
            u = c**(1 - self.sigma) / (1 - self.sigma)
        return u

    def interest(self, k):
        return self.theta * (k / self.l_s)**(self.theta - 1) - self.delta

    def interest_reverse(self, r):
        return (self.theta / (r + self.delta)
                )**(1 / (1 - self.theta)) * self.l_s

    def r_to_w(self, r):
        return (1 - self.theta) * ((self.theta / (r + self.delta))
                                   )**(self.theta / (1 - self.theta))

    def markov(self):
        self.mc = qe.markov.approximation.tauchen(self.rho, self.stdz,
                                                  0, self.m, self.nz)
        self.P = self.mc.P
        self.l = np.exp(self.mc.state_values)
        inv_l = np.linalg.matrix_power(self.P, 1000)
        inv_dist = inv_l[0, :]
        self.l_s = np.dot(self.l, inv_dist)
        return self.P, self.l_s


# Generating a class
nz = 7
nk = 1000
sigma = 3
rho = 0.6
hh = HH(nz=nz, nk=nk, rho=rho, sigma=sigma)


# Current level
P, l_s = hh.markov()
r = (3.87 - 1) / 100
k_t = hh.interest_reverse(r)


# Extrcting a policy function, given a Value function
@jit
def Value_max(V, r, HH):
    # Unpacking of parameters
    nz, nk, P = HH.nz, HH.nk, HH.P
    l, k = HH.l, HH.k
    beta = HH.beta
    R = 1 + r
    w = HH.r_to_w(r)

    # Empty matrices
    g = np.zeros((nz, nk))
    indk = np.copy(V)
    for iz in range(nz):
        for ik in range(nk):
            c = w * l[iz] + R * k[ik] - k
            u = HH.utility(c)
            u[c < 0] = -1000000
            v = u + beta * (np.dot(P[iz, :], V))
            ind = np.argmax(v)
            indk[iz, ik] = ind
            g[iz, ik] = k[ind]
    return g, indk


# Function to setup J given a guessed policy function
@jit
def J(g, HH):
    k, nz, nk = HH.k, HH.nz, HH.nk
    J = np.zeros((nz, nk, nk))
    for i in range(nz):
        for j in range(nk):
            J[i, j, :] = (g[i, j] == k)
    return J


# Function to setup a similar matrix as the Q matrix to Ljundgqvist and
# Sargent:
@jit
def Q(P, J):
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
def Value(Q, re, HH):
    nz, nk, beta = HH.nz, HH.nk, HH.beta
    matrix = np.eye(np.shape(Q)[0]) - beta * Q
    inverse = np.linalg.inv(matrix)
    v = np.dot(inverse, re)
    v = v.reshape(nz, nk)
    return v


# Generate a reward vector
@jit
def reward(r, HH, g):
    nk, nz, k, l = HH.nk, HH.nz, HH.k, HH.l
    w = hh.r_to_w(r)
    re = hh.utility((1 + r) * k.reshape(1, nk) + w * l.reshape(nz, 1) - g)
    return re.flatten()


# Policy function iteration
def policy(g, r, HH, maxiter=1000, tol=10**(-11)):
    error = 1
    iter = 0
    test1 = (error > tol)
    test2 = (iter < maxiter)
    while (test1 and test2):
        # Generate J
        j = J(g, HH)

        # Generate Q
        q = Q(HH.P, j)

        # Getting a reward vector
        re = reward(r, HH, g)

        # Generate a new value function
        v = Value(q, re, HH)

        # Extract a policy function
        gnew, ink = Value_max(v, r, HH)

        error = np.linalg.norm(gnew - g)
        g = np.copy(gnew)
        iter = iter + 1
        test1 = (error > tol)
        test2 = (iter < maxiter)
    return g, v, ink, j


# Calculating the invariate distribution
@jit
def distribution(indk, HH, tol=10**(-11), maxiter=10000):
    nz, nk = HH.nz, HH.nk
    dist = np.ones((nz, nk)) / (nz * nk)

    error = 1
    iter = 0
    test1 = (error > tol)
    test2 = (iter < maxiter)
    while (test1 and test2):
        distnew = np.zeros((nz, nk))
        for j in range(nk):
            for i in range(nz):
                distnew[:, int(indk[i, j])] = distnew[:, int(
                    indk[i, j])] + dist[i, j] * P[i, :]
        error = np.linalg.norm(distnew - dist)
        dist = np.copy(distnew)
        test1 = (error > tol)
        test2 = (iter < maxiter)
        iter = iter + 1
    return dist


# Function to solve for the equilibrium
@jit
def Aiyagari(k_t, HH):
    # Unpacking parameters
    beta, theta, delta = HH.beta, HH.theta, HH.delta
    l_s = HH.l_s

    iter = 0
    error = 10
    tol = 0.01
    g = np.zeros((nz, nk))
    test = (error > tol)
    while test:
        r = HH.interest(k_t)
        if r > 1 / beta - 1:
            r = 1 / beta - 1
        iter = iter + 1

        # Value function iteration
        start1 = time.time()
        g, V, indk, j = policy(g, r, hh)
        stop1 = time.time()

        # Find capital supply
        start2 = time.time()
        dist = distribution(indk, hh)
        k_s = np.sum(hh.k * np.sum(dist, axis=0))
        stop2 = time.time()
        r1 = theta * (k_s / l_s)**(theta - 1) - delta
        error = np.abs(r - r1) * 100
        print("\n------------------------------------------------------------")
        print("The error in iteration %.0F is %F." % (iter, error))
        print("The capital supply is %F, and the interest rate is %F." %
              (k_s, r * 100))
        print("PFI and simulation took %.3F and %.3F seconds respectively" %
              ((stop1 - start1), (stop2 - start2)))

        # To make convergence easier, we altered the updating function
        # the closer we get to our accepted tolerance level
        if error > 2:
            k_t = 0.95 * k_t + 0.05 * k_s
        elif error > 0.5:
            k_t = 0.99 * k_t + 0.01 * k_s
        elif error > 0.05:
            k_t = 0.995 * k_t + 0.005 * k_s
        elif error > 0.02:
            k_t = 0.9975 * k_t + 0.0025 * k_s
        else:
            k_t = 0.999 * k_t + 0.001 * k_s
        test = (error > tol)
    print("\nThe equilibrium interest rate is %F." % (r * 100))
    # Das Ziel sollte 3.87 sein
    # Resultat war 3.6498 (Check against QE results)
    return V, g, indk, dist, r


# Running the function
start = time.time()
V, g, indk, dist, r = Aiyagari(k_t, hh)
stop = time.time()
print("Solving the model took %F minutes." % ((stop - start) / 60))


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
def simulate(mc, indk, l=sim, k=k, N=10000, T=T):
    nz = np.shape(mc.P)[0]
    T = np.shape(l)[0]
    m = T / N
    ind = np.zeros(N)
    for n in range(N):
        l = np.concatenate((l[int(n * m):T], l[0:int(n * m)]))
        temp = indk[int((nz - 1) / 2), int(nk / 2)]
        for t in range(T):
            temp = indk[int(l[t]), int(temp)]
        ind[n] = temp
    a = k[np.int64(ind)]
    return a


# Generate a new distribution
dist2 = simulate(mc, indk, N=10000)


# Plot the distribution
plt.figure(figsize=(15, 10))
n, bins, patches = plt.hist(
    x=dist2, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, histtype="stepfilled")
plt.xlabel('Asset Value')
plt.ylabel('Frequency')
plt.title('Asset Distribution')
plt.show()
plt.savefig("distribution.png")


# Print the output
print("\nThe equilibrium interest rate is %F." % (r * 100))
print("Solving the model took %F minutes." % ((stop - start) / 60))
print("The gini coefficient for the distribution is %F." %
      (qe.gini_coefficient(dist2)))
