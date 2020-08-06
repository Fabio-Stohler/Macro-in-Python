import autograd.numpy as np
from autograd import jacobian

np.set_printoptions(suppress=True,precision=4)



# Parameters and setup
beta = 0.97
gamma = 2.0
rho = 0.95
rho_xi = 0.80
zeta = 0.8
psi = 0.1
mu = 1.2  # markup  = epsilon / (epsilon - 1)
mu_epsilon = mu/(mu-1)
theta = 0.75
omega = 1.5
ubar = 0.06
delta = 0.15 # job separation prob
Mbar = (1-ubar)*delta/(ubar+delta*(1-ubar))
wbar = 1/mu - delta * psi * Mbar
B = 0.6
ben = wbar * 0.5

amin = 0.0  # borrowing constraint
#exogenous transition matrix
#has form rows = future state, columns = current  state
N = 2

#grid for savings
Amin, Amax, Asize = amin, 200, 201
A = np.linspace(Amin**(0.25), Amax**(0.25), Asize)**4

#prepare some useful arrays
tiledGrid = np.tile(A,(2,1))




# The Prices class wraps the items needed as inputs to the household and firm problems
class Prices:
    def __init__(self,R,M=Mbar,Z=1.0,u=ubar,ulag=ubar):
        self.R = R
        self.M = M
        self.w = wbar * (M/Mbar)**zeta
        Y = Z * (1-u)
        H = (1-u) - (1-delta)*(1-ulag)
        d = (Y-psi*M*H)/(1-u) - self.w
        tau = ((self.R-1)*B + ben * u)/(self.w+d)/(1-u)
        self.employedIncome = (1-tau)*(self.w+d)
        self.earnings = np.array([ben,self.employedIncome])
        self.employTrans = np.array([[1-M,delta*(1-M)],[M,1.0-delta*(1-M)]])

    def tiledEndow(self):
        return np.tile(self.earnings[np.newaxis].T,(1,Asize))

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
    return np.vstack( [Pr.R * CurrentAssets[i] + Pr.earnings[i] - interp(G[i],A,CurrentAssets[i])[0] for i in range(N)] )


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
            Eupcp_i += Pr_P.employTrans[jp,ip] * upcp[jp]
        Eupcp.append(Eupcp_i)
    Eupcp = np.vstack(Eupcp)

    #use  upc = R *  beta*Eupcp to solve for upc
    upc = beta*Pr_P.R*Eupcp

    #invert uprime to solve for c
    c = uPrimeInv(upc)


    #use budget constraint to find previous assets
    # (a' + c - y)/R = a
    a = (tiledGrid + c - Pr.tiledEndow())/ Pr.R

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
G = 10+tiledGrid
R = 1.01
Pr0 = Prices(R)

# This material below runs only if this file is executed as a script
if __name__ == "__main__":
    G = SolveEGM(G,Pr0)
    import matplotlib.pyplot as plt
    lines = plt.plot(A[:30],get_c(G,Pr0)[:,:30].T)
    plt.xlabel('Current Assets')
    plt.ylabel('Consumption')
    plt.legend(lines, ['Low endow', 'High endow'])
