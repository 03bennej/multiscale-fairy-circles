import cupy as cp
from cupy.fft import fft2, ifft2


def get_kin(par):
    @cp.fuse(kernel_name='kin')
    def kin(u1, v1, u2, v2):
        """
        Define non spatial parts of equations
        """

        f1 = par['lam1']*u1*par['n1']*par['Z1']*v1*(1+par['e1']*u1)**2*(1-u1/par['k1'])-par['m1']*u1 \
            -(par['a']*u1*v1*v2*(v2-v1))/((v1+par['s1star'])*(v2+par['s2star']))
        
        g1 = par['p']/(par['n1']*par['Z1']) - par['N']*v1/(1+par['r1']*u1/par['k1']+par['r2']*u2/par['k2']) \
            - par['gam1']*u1*v1*(1+par['e1']*u1)**2 - par['theta1']*par['gam2']*u2*v1*(1+par['e2']*u2)**2 \
                - par['beta1']*v1**2/(par['n1']*par['Z1'])
        
        f2 = (par['theta1']*par['n1']*par['Z1']*v1+par['theta2']*par['n2']*par['Z2']*v2)*par['lam2']*u2*(1+par['e2']*u2)**2*(1-u2/par['k2']) \
            -par['m2']*u2 +(par['a']*u1*v1*v2*(v2-v1))/((v1+par['s1star'])*(v2+par['s2star']))
        
        g2 = (par['beta1']*v1**2 - par['beta2']*v2**2)/(par['n2']*par['Z2']) - par['theta2']*par['gam2']*u2*v2*(1+par['e2']*u2)**2 
        
        return f1, g1, f2, g2
    return kin

# define full system
def get_ku1in(par):
    @cp.fuse(kernel_name='ku1in')
    def ku1in(u1, v1, u2, v2, phiu1, phiu2):  
        """
        Convolution terms are calculated via Fourier transform but cp.fuse() does not support fft,
        hence spatial terms are passed as arguments to be calculated outside this function
        as phiu1 and phiu2.
        """
        kin = get_kin(par)
        ku1in=kin(u1, v1, u2, v2)[0] \
        - par['fe1']*u1+par['db1']*cp.real(phiu1)*(par['Rmax']*(u1+u2)+par['br']*par['Rmin'])/(u1+u2+par['br']) \
        + par['db2']*cp.real(phiu2)*(par['Rmax']*(u1+u2)+par['br']*par['Rmin'])/(u1+u2+par['br'])
        return ku1in
    return ku1in

def get_kv1in(par):
    @cp.fuse(kernel_name='kv1in')
    def kv1in(u1, v1, u2, v2):
        kin = get_kin(par)
        kv1in=kin(u1, v1, u2, v2)[1]
        return kv1in
    return kv1in

def get_ku2in(par):
    @cp.fuse(kernel_name='ku2in')
    def ku2in(u1, v1, u2, v2):
        kin = get_kin(par)
        ku2in=kin(u1, v1, u2, v2)[2] - par['fe2']*u2
        return ku2in
    return ku2in

def get_kv2in(par):
    @cp.fuse(kernel_name='kv2in')
    def kv2in(u1, v1, u2, v2):
        kin = get_kin(par)
        kv2in=kin(u1, v1, u2, v2)[3]
        return kv2in
    return kv2in

def solver(u1, v1, u2, v2, par, ksq, dt):
    """
    Takes u1, v1, u2, v2 and returns solution at t+dt
    """

    # get fused GPU functions for the given parameter set
    ku1in = get_ku1in(par)
    kv1in = get_kv1in(par)
    ku2in = get_ku2in(par)
    kv2in = get_kv2in(par)

    # Apply Fourier transforms 
    u1hat=fft2(u1)
    v1hat=fft2(v1)
    u2hat=fft2(u2)
    v2hat=fft2(v2)

    ############# Runge Kutta fourth order ##############
    phiu1 = ifft2(cp.exp(-par['l1']**2*ksq/2)*u1hat)
    phiu2 = ifft2(cp.exp(-par['l2']**2*ksq/2)*u2hat)
    ku11 = dt*fft2(ku1in(u1, v1, u2, v2, phiu1, phiu2))
    del phiu1,phiu2
    ku21 =dt*fft2(ku2in(u1, v1, u2, v2))
    kv11 =dt*fft2(kv1in(u1, v1, u2, v2))
    kv21 =dt*fft2(kv2in(u1, v1, u2, v2))
    del u1,v1,u2,v2
    Eu1=1
    u12=cp.real(ifft2(Eu1*(u1hat+ku11/2)))
    u1=Eu1**2*u1hat+(Eu1**2*ku11)/6
    del Eu1,ku11
    Ev1=cp.exp(-par['ds1']*dt*ksq/2)
    v12=cp.real(ifft2(Ev1*(v1hat+kv11/2)))
    v1=Ev1**2*v1hat+(Ev1**2*kv11)/6
    del Ev1,kv11
    Eu2=cp.exp(-par['dr']*dt*ksq/2)
    u22=cp.real(ifft2(Eu2*(u2hat+ku21/2)))
    u2=Eu2**2*u2hat+(Eu2**2*ku21)/6
    del Eu2,ku21
    Ev2=cp.exp(-par['ds2']*dt*ksq/2)
    v22=cp.real(ifft2(Ev2*(v2hat+kv21/2))) 
    v2=Ev2**2*v2hat+(Ev2**2*kv21)/6 
    del Ev2,kv21
    
    phiu1 = ifft2(cp.exp(-par['l1']**2*ksq/2)*fft2(u12))
    phiu2 = ifft2(cp.exp(-par['l2']**2*ksq/2)*fft2(u22))
    ku12 =dt*fft2(ku1in(u12, v12, u22, v22, phiu1, phiu2))
    del phiu1,phiu2
    ku22 =dt*fft2(ku2in(u12, v12, u22, v22))
    kv12 =dt*fft2(kv1in(u12, v12, u22, v22))
    kv22 =dt*fft2(kv2in(u12, v12, u22, v22))
    del u12,v12,u22,v22
    Eu1=1
    u13=cp.real(ifft2(Eu1*u1hat+ku12/2))
    u1=u1+(2*Eu1*ku12)/6
    del Eu1,ku12    
    Ev1=cp.exp(-par['ds1']*dt*ksq/2)
    v13=cp.real(ifft2(Ev1*v1hat+kv12/2))
    v1=v1+(2*Ev1*kv12)/6
    del Ev1,kv12
    Eu2=cp.exp(-par['dr']*dt*ksq/2)
    u23=cp.real(ifft2(Eu2*u2hat+ku22/2))
    u2=u2+(2*Eu2*ku22)/6
    del Eu2,ku22
    Ev2=cp.exp(-par['ds2']*dt*ksq/2)
    v23=cp.real(ifft2(Ev2*v2hat+kv22/2)) 
    v2=v2+(2*Ev2*kv22)/6  
    del Ev2,kv22

    phiu1 = ifft2(cp.exp(-par['l1']**2*ksq/2)*fft2(u13))
    phiu2 = ifft2(cp.exp(-par['l2']**2*ksq/2)*fft2(u23))
    ku13 =dt*fft2(ku1in(u13, v13, u23, v23, phiu1, phiu2))
    del phiu1,phiu2
    ku23 =dt*fft2(ku2in(u13, v13, u23, v23))
    kv13 =dt*fft2(kv1in(u13, v13, u23, v23))
    kv23 =dt*fft2(kv2in(u13, v13, u23, v23))
    del u13,v13,u23,v23
    Eu1=1
    u14 = cp.real(ifft2(Eu1**2*u1hat+Eu1*ku13))
    u1=u1+(2*Eu1*ku13)/6
    del Eu1,ku13
    Ev1=cp.exp(-par['ds1']*dt*ksq/2)
    v14 = cp.real(ifft2(Ev1**2*v1hat+Ev1*kv13))
    v1=v1+(2*Ev1*kv13)/6
    del Ev1,kv13
    Eu2=cp.exp(-par['dr']*dt*ksq/2)
    u24 = cp.real(ifft2(Eu2**2*u2hat+Eu2*ku23))
    u2=u2+(2*Eu2*ku23)/6
    del Eu2,ku23
    Ev2=cp.exp(-par['ds2']*dt*ksq/2)    
    v24 = cp.real(ifft2(Ev2**2*v2hat+Ev2*kv23)) 
    v2=v2+(2*Ev2*kv23)/6
    del Ev2,kv23
    
    phiu1 = ifft2(cp.exp(-par['l1']**2*ksq/2)*fft2(u14))
    phiu2 = ifft2(cp.exp(-par['l2']**2*ksq/2)*fft2(u24))
    ku14 =dt*fft2(ku1in(u14, v14, u24, v24, phiu1, phiu2))
    u1=cp.real(ifft2(u1+ku14/6))
    del phiu1,phiu2
    ku24 =dt*fft2(ku2in(u14, v14, u24, v24))
    u2=cp.real(ifft2(u2+ku24/6))
    del ku24
    kv14 =dt*fft2(kv1in(u14, v14, u24, v24))
    v1=cp.real(ifft2(v1+kv14/6))
    del kv14
    kv24 =dt*fft2(kv2in(u14, v14, u24, v24))
    v2=cp.real(ifft2(v2+kv24/6))
    del kv24
    ###########################################################

    return u1, v1, u2, v2