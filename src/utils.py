import cupy as cp
from numpy import linalg as LA
import numpy as np
import pickle
import os

def save(path, u1, v1, u2, v2, par):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as file:
        pickle.dump({
                    'u1': cp.asnumpy(u1),
                    'v1': cp.asnumpy(v1),
                    'u2': cp.asnumpy(u2),
                    'v2': cp.asnumpy(v2),
                    'params': par,             
                    }, 
                    file)
        
def load(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def non_spatial_parts(y, par):  # non-spatial parts of the integro-pde system
        
        u1, v1, u2, v2 = y

        f1 = par['lam1']*u1*par['n1']*par['Z1']*v1*(1+par['e1']*u1)**2*(1 - u1/par['k1'])-par['m1']*u1 - (par['a']*u1*v1*v2*(v2-v1))/((v1+par['s1star'])*(v2+par['s2star']))
        
        g1 = par['p']/(par['n1']*par['Z1']) - par['N']*v1/(1+par['r1']*u1/par['k1']+par['r2']*u2/par['k2']) \
            - par['gam1']*u1*v1*(1+par['e1']*u1)**2 - par['theta1']*par['gam2']*u2*v1*(1+par['e2']*u2)**2 \
                - par['beta1']*v1**2/(par['n1']*par['Z1'])
        
        f2 = (par['theta1']*par['n1']*par['Z1']*v1+par['theta2']*par['n2']*par['Z2']*v2)*par['lam2']*u2*(1+par['e2']*u2)**2*(1-u2/par['k2']) \
            - par['m2']*u2 +(par['a']*u1*v1*v2*(v2-v1))/((v1+par['s1star'])*(v2+par['s2star']))
        
        g2 = (par['beta1']*v1**2 - par['beta2']*v2**2)/(par['n2']*par['Z2']) - par['theta2']*par['gam2']*u2*v2*(1+par['e2']*u2)**2 

        F = [f1, g1, f2, g2]

        return F


def homogeneous_system(y, par):  # homogeneous system including contributions from spatial terms

    u1, v1, u2, v2 = y

    f1, g1, f2, g2 = non_spatial_parts(y, par)

    du1dt = f1 \
    - par['fe1']*u1+(par['db1']*u1+par['db2']*u2)*(par['Rmax']*(u1+u2) + par['br']*par['Rmin'])/(u1+u2+par['br'])
    dv1dt = g1
    du2dt = f2 - par['fe2']*u2
    dv2dt = g2

    dydt = [du1dt, dv1dt, du2dt, dv2dt]

    return dydt

def get_homogeneous_system(par):  # wrapper to generate function for odeint
    return lambda y, t: homogeneous_system(y, par)

def DD(k, steady_state, par):

    b1, s1, b2, s2 = steady_state

    phi1=np.exp(-(par['l1']*k)**2/2)
    phi2=np.exp(-(par['l2']*k)**2/2)

    Dmat=np.zeros([4,4])
    
    Dmat[0,0] = par['db1'] * (par['Rmax'] * b1 ** 2 * phi1 + 2 * par['Rmax'] * b1 * b2 * phi1 + par['Rmax'] * b1 * par['br'] * phi1 + par['Rmax'] * b2 ** 2 * phi1 + par['Rmax'] * b2 * par['br'] * phi1 + par['Rmin'] * b1 * par['br'] * phi1 + par['Rmin'] * b2 * par['br'] * phi1 + par['Rmin'] * par['br'] ** 2 * phi1 + par['br'] * b1 * par['Rmax'] - par['br'] * b1 * par['Rmin']) / (b1 + b2 + par['br']) ** 2 + par['db2'] * par['br'] * (par['Rmax'] - par['Rmin']) * b2 / (b1 + b2 + par['br']) ** 2 - par['fe1']
    Dmat[0,2] = par['db1'] * par['br'] * (par['Rmax'] - par['Rmin']) * b1 / (b1 + b2 + par['br']) ** 2 + par['db2'] * (par['Rmax'] * b1 ** 2 * phi2 + 2 * par['Rmax'] * b1 * b2 * phi2 + par['Rmax'] * b1 * par['br'] * phi2 + par['Rmax'] * b2 ** 2 * phi2 + par['Rmax'] * b2 * par['br'] * phi2 + par['Rmin'] * b1 * par['br'] * phi2 + par['Rmin'] * b2 * par['br'] * phi2 + par['Rmin'] * par['br'] ** 2 * phi2 + par['Rmax'] * b2 * par['br'] - par['Rmin'] * b2 * par['br']) / (b1 + b2 + par['br']) ** 2
    Dmat[1,1] = - par['ds1'] * k**2
    Dmat[2,2] = - par['dr'] * k**2 - par['fe2']
    Dmat[3,3] = - par['ds2'] * k**2

    return Dmat

def max_real_eigenvalue(J0, steady_state, par, k):

    maxrelam = np.zeros(np.shape(k)[0])
    max_evectors = np.zeros((np.shape(k)[0], 4))
    flags = np.zeros(np.shape(k)[0])  # flags to indicate complex number
    n = np.shape(k)[0]

    for i in range(n):
        Dmat = DD(k[i], steady_state, par)
        reroots, evectors = LA.eig(J0 + Dmat)
        j = np.argsort(reroots)
        reroots = reroots[j]
        evectors = evectors[:, j]
        if not reroots.any():
             maxrelam[i] = np.nan
             max_evectors[i] = np.nan
        else:
             jj = np.argmax(np.real(reroots))
             maxrelam[i] = np.max(np.real(reroots))
             max_evectors[i, :] = evectors[:, jj]
             flags[i] = np.isclose(reroots[-1].imag, 0)

    return k, maxrelam, flags, max_evectors