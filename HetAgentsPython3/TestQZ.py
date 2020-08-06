import numpy as np
import scipy.linalg as sl

import warnings

from HANK_Reiter import AMat, BMat, CMat, EMat


import matlab.engine
from matlab import double as matdouble

import warnings

eng = matlab.engine.start_matlab()
eng.addpath('../Matlab')

def MConv(x):
    return matdouble(x.tolist())



HasLead = np.any(np.abs(AMat) > 1e-9,axis = 1)

Ashift = AMat.copy()
Bshift = BMat.copy()
Cshift = CMat.copy()

Ashift[~HasLead,:] = BMat[~HasLead,:]
Bshift[~HasLead,:] = CMat[~HasLead,:]
Cshift[~HasLead,:] = 0.0


IsLag = np.where(np.any(np.abs(Cshift) > 1e-9,axis = 0))[0] # indices of variables that need auxiliaries

n = AMat.shape[0]
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

nE = EMat.shape[1]
E = np.vstack((EMat,np.zeros((naux,nE))))

psi = E
pi = EE



Lambda,Omega, alpha, beta,q,z = sl.ordqz(G,H,output='complex',sort = 'ouc')
q = q.conj().T
# note q.T @ Lambda @ z.T - G = 0 (see eq 4.49 in DeJong and Dave)

g0 = G.copy()
g1 = H.copy()
# Lambda, Omega, q, z = [np.array(xx) for xx in eng.testordqz(MConv(g0),MConv(g1),nargout = 4)]


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

Phi = q1pi @ np.linalg.inv(q2pi)

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



print('Now matlab')
eu,eigvals,mG1,mimpact =[ np.array(xx) for xx in eng.checkeu(MConv(g0),MConv(g1),MConv(psi),MConv(pi),1+1e-10,nargout =4 )]
