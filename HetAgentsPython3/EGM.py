import autograd.numpy as np
from autograd import jacobian
np.set_printoptions(suppress=True,precision=4)
from Support import GetStationaryDist

from RBC import alpha, beta, delta, gamma, rho


amin = 0.0  # borrowing constraint
#exogenous transition matrix
#has form rows = future state, columns = current  state
N = 2
lamw = 0.6 # job finding prob
sigma = 0.2 # job separation prob
exogTrans = np.mat([[1-lamw,sigma],[lamw,1.0-sigma]])

#labor endowments
endow = np.array([1.0,2.5])
Lbar = np.dot(endow,GetStationaryDist(exogTrans))
endow = endow / Lbar
Lbar = 1.0

#grid for savings
gridmin, gridmax, gridsize = amin, 200, 201
grid = np.linspace(gridmin**(0.25), gridmax**(0.25), gridsize)**4

#prepare some useful arrays
tiledGrid = np.tile(grid,(2,1))
tiledEndow = np.tile(endow[np.newaxis].T,(1,gridsize))

# code firm focs for prices
class Prices:
    def __init__(self,K,Z = 1.0):
        self.R = Z * alpha*(K/Lbar)**(alpha-1) + 1 - delta
        self.w = Z * (1-alpha)*(K/Lbar)**(alpha)

def u(c):
    return (c**(1-gamma)-1.0)/(1.0-gamma)

def uPrime(c):
    return c**(-gamma)

def uPrimeInv(up):
    return up**(-1.0/gamma)

def interp(x,y,x1):
    N = len(x)
    i = np.minimum(np.maximum(np.searchsorted(x,x1,side='right'),1),N-1)
    xl = x[i-1]
    xr = x[i]
    yl = y[i-1]
    yr = y[i]
    y1 = yl + (yr-yl)/(xr-xl) * (x1-xl)
    above = x1 > x[-1]
    below = x1 < x[0]
    y1 = np.where(above,y[-1] +   (x1 - x[-1]) * (y[-1]-y[-2])/(x[-1]-x[-2]), y1)
    y1 = np.where(below,y[0],y1)

    return y1, i


def get_c(G,Pr,CurrentAssets = tiledGrid):
    return np.vstack( [Pr.R * CurrentAssets[i] + Pr.w*endow[i] - interp(G[i],grid,CurrentAssets[i])[0] for i in range(N)] )


def eulerBack(G,Pr,Pr_P):
# The argument is the savings policy rule in the next period
# it is parameterized as the a values associated with savings grid for a'

    # compute next period's consumption conditional on next period's income
    cp = get_c(G,Pr_P)
    upcp = uPrime(cp)
    #compute E(u'(cp))
    # In principle we could do it like this: Eupcp = np.dot(exogTrans.T , uPrime(cp) )
    # But automatic differentiation doesnt work with matrix-matrix multiplication
    # because it isn't obvious how to take the gradient of a function that produces a matrix
    # so we loop over the values instead
    Eupcp = []
    for ip in range(N):
        Eupcp_i = 0.
        for jp in range(N):
            Eupcp_i += exogTrans[jp,ip] * upcp[jp]
        Eupcp.append(Eupcp_i)
    Eupcp = np.vstack(Eupcp)

    #use  upc = R' *  beta*Eupcp to solve for upc
    upc = beta*Pr_P.R*Eupcp

    #invert uprime to solve for c
    c = uPrimeInv(upc)

    #use budget constraint to find previous assets
    # (a' + c - y)/R = a
    a = (tiledGrid + c - Pr.w*tiledEndow)/ Pr.R

    return a, c



def SolveEGM(G,Pr):
    #loop until convergence
    print("solving for policy rules")
    tol = 1e-15
    test = True
    for it in range(10000):
        a = eulerBack(G,Pr,Pr)[0]

        if it % 50 == 0:
            test = np.abs(a-G)/(np.abs(a)+np.abs(G)+tol)
            print("it = {0}, test = {1}".format(it,test.max()))
            if np.all(test  < tol):
                break

        G = a

    return G

#initialize
G = 10+0.1*tiledGrid
Pr0 = Prices(48.0)

# This material below runs only if this file is executed as a script
if __name__ == "__main__":
    G = SolveEGM(G,Pr0)
    import matplotlib.pyplot as plt
    lines = plt.plot(grid,get_c(G,Pr0).T)
    plt.xlabel('Current Assets')
    plt.ylabel('Consumption')
    plt.legend(lines, ['Low endow', 'High endow'])
    plt.show()

    plt.figure()
    Ln = plt.plot(grid,interp(G[0],grid,grid)[0])
    Ln.append(plt.plot(grid,interp(G[1],grid,grid)[0])[0])
    l = plt.plot(grid,grid)[0]
    l.set_linestyle('--')
    l.set_color([0.6,0.6,0.6])
    plt.xlabel('Current Assets')
    plt.ylabel('Savings')
    plt.legend(Ln, ['Low endow', 'High endow'])
    plt.xlim([0,50])
    plt.ylim([0,50])
    plt.show()
