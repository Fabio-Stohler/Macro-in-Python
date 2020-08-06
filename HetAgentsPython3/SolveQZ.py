import numpy as np
import scipy.linalg as sl
import warnings

def checkeu(g0,g1,psi,pi):
    div = 1. + 1e-10

    Lambda,Omega, alpha, beta,q,z = sl.ordqz(g0,g1,output='complex',sort = 'ouc')
    q = q.conj().T

    # partition into explosive and non-explosive blocks
    dLambda = np.abs(np.diag(Lambda))
    dOmega = np.abs(np.diag(Omega))
    dLambda = np.maximum(dLambda,1e-10) #to avoid dividing by 0;
    n = Lambda.shape[0]
    n_unstable = np.sum(np.logical_or(dLambda<=1e-10 , dOmega/dLambda>(1+1e-10)))
    n_stable = n-n_unstable



    iStable = np.arange(n-n_unstable)
    iUnstable = np.arange(n-n_unstable,n)

    q1=q[iStable,:]
    q2=q[iUnstable,:]
    q1pi=q1@pi
    q2pi=q2@pi
    q2psi=q2@psi

    iExist = np.linalg.matrix_rank(q2pi) == np.linalg.matrix_rank(np.hstack((q2psi,q2pi)))
    iUnique = np.linalg.matrix_rank(q@pi) == np.linalg.matrix_rank(q2pi)
    eu = np.hstack((iExist, iUnique))

    Phi = q1pi @ np.linalg.inv(q2pi)  # is this right? is q2pi square?  Sims uses SVD

    z1 = z[:,iStable]

    L11 = Lambda[np.ix_(iStable,iStable)]
    L12 = Lambda[np.ix_(iStable,iUnstable)]
    L22 = Lambda[np.ix_(iUnstable,iUnstable)]

    O11 = Omega[np.ix_(iStable,iStable)]
    O12 = Omega[np.ix_(iStable,iUnstable)]
    O22 = Omega[np.ix_(iUnstable,iUnstable)]



    L11inv = np.linalg.inv(L11)

    aux = np.hstack((O11, O12-Phi@O22  )) @ z.conj().T

    aux2 = z1@L11inv

    G1 = np.real(aux2@aux)

    aux = np.vstack((  np.hstack((L11inv, -L11inv@(L12-Phi@L22))),
        np.hstack((np.zeros((n_unstable,n_stable)), np.eye(n_unstable) ))
        ))


    H = z@aux

    impact = np.real(H@np.vstack((q1-Phi@q2,np.zeros((n_unstable,psi.shape[0]))))@psi)

    return eu, G1, impact


def solve(A,B,C,E):

    HasLead = np.any(np.abs(A) > 1e-9,axis = 1)

    Ashift = A.copy()
    Bshift = B.copy()
    Cshift = C.copy()

    Ashift[~HasLead,:] = B[~HasLead,:]
    Bshift[~HasLead,:] = C[~HasLead,:]
    Cshift[~HasLead,:] = 0.0


    IsLag = np.where(np.any(np.abs(Cshift) > 1e-9,axis = 0))[0] # indices of variables that need auxiliaries

    n = A.shape[0]
    naux = IsLag.size
    iaux = range(n,n+naux)  # indices of corresponding auxiliary variables

    G = np.zeros((n+naux,n+naux))
    H = np.zeros((n+naux,n+naux))

    G[:n,:n] = -Ashift
    H[:n,:n] = Bshift
    H[:n,iaux] = Cshift[:,IsLag]

    G[iaux,iaux] = 1.0
    H[iaux,IsLag] = 1.0




    nEE = np.nonzero(HasLead)[0].size
    EE = np.zeros((n+naux,nEE))
    EE[np.where(HasLead)[0],range(nEE)] = 1.0

    nE = E.shape[1]
    E = np.vstack((E,np.zeros((naux,nE))))


    eu, G1, impact  = checkeu(G,H,E,EE)

    if not all(eu):
        print("eu = {}".format(eu.flatten()))
        raise RuntimeError("existence or uniqueness problem.")

    assert np.max(np.abs(G1[:,-naux:])) < 1e-6
    G1 = G1[:-naux,:-naux]
    impact = impact[:-naux,:]

    test = np.max(np.abs(C+B@G1+A@G1@G1))
    if test > 1e-6:
        print("test residual = {}".format(test))
        warnings.warn("Theoretical check in SolveQZ.solve fails.")

    return G1, impact
