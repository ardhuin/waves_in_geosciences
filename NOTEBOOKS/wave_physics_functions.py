# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import numpy as np
import sys
from scipy.interpolate import griddata
###################### 
######################  
# [Ekxky,kx,ky,kx2,ky2] = wavespec_Efth_to_Ekxky(eft1s,fren,dfreq,dirn,dth,dkx=0.0001,dky=0.0001,nkx=250,nky=250,doublesided=0) 
def  wavespec_Efth_to_Ekxky(eft1s,fren,dfreq,dirn,dth,depth=3000.,dkx=0.0001,dky=0.0001,nkx=250,nky=250,doublesided=1,verbose=0,doplot=0,trackangle=0)  :
    '''
    Converts E(f,theta) spectrum from buoy or model to E(kx,ky) spectrum similar to image spectrum
    2023/11/14: preliminary version, assumes dfreq is symmetric (not eaxctly true with WW3 output and waverider data) 
    inputs :
            - etfs1 : spectrum
    output : 
            - Ekxky: spectrum
            - kx: wavenumber in cycles / m  
    '''
    [nf,nt]=np.shape(eft1s)
    tpi=2*np.pi
    grav=9.81

# makes a double sided spectrum
    if doublesided == 1:
        eftn=0.5*(eft1s+np.roll(eft1s,nt//2,axis=1))
    else: 
        eftn=eft1s
    Hs1 = 4*np.sqrt(np.sum(np.sum(eftn,axis=1)* dfreq)*dth)
# wraps around directions
    dlast=dirn[0]+360.
    dirm=np.concatenate([dirn,[dlast]])
    elast=eftn[:,0]
    eftm1=np.concatenate([eftn.T,[elast]]).T
# adds zero energy in a low frequency to avoid interpolation across k=0
    ffirst=fren[0]-0.9*(fren[1]-fren[0])
    frem=np.concatenate([[ffirst],fren])
    efirst=eftm1[0,:]*0
    eftm=np.concatenate([[efirst],eftm1])

#plt.pcolormesh(fren, dirm, np.log10(eftm).T)
    km=(2*np.pi*frem)**2/(grav*2*np.pi)   # cycles / meter
    for ii in range(nf):
       km[ii]=k_from_f(frem[ii],D=depth)/(2*np.pi)            # finite water depth
    km2=np.tile(km.reshape(nf+1,1),(1,nt+1))

# eftn*df*dth = Ek*k*dk*dth -> Ek = efth *df /(k * dk)  =  efth *Cg /k
    Cg2=np.sqrt(grav/(km2*tpi))*0.5
    Jac=Cg2/km2
    dirm2=np.tile(dirm.T,(nf+1,1))*np.pi/180.
    kxn=km2*np.cos(dirm2+trackangle)
    kyn=km2*np.sin(dirm2+trackangle)
    #plt.scatter(kxn,kyn,  marker='.', s = 20)
    kx=np.linspace(-nkx*dkx,(nkx-1)*dkx,nkx*2)
    ky=np.linspace(-nky*dky,(nky-1)*dky,nky*2)
    kx2, ky2 = np.meshgrid(kx,ky,indexing='ij')   #should we transpose kx2 and ky2 ???
    Ekxky = griddata((kxn.flatten(), kyn.flatten()), (eftm*Jac).flatten(), (kx2, ky2), method='nearest')
    Hs2=4*np.sqrt(np.sum(np.sum(Ekxky))*dkx*dky)
# make sure energy is exactly conserved (assuming kmax is consistent with fmax
    if verbose==1: 
        print('Hs1,Hs2:',Hs1,Hs2)
    Ekxky = Ekxky * (Hs1/Hs2)**2
    return Ekxky,kx,ky,kx2,ky2

#############################################################################
def  wavespec_Efth_to_first3(efth,fren,dfreq,dirn,dth,cut=1E4)  :
    '''
    Computes first 3 moments from E(f,theta) spectrum
    inputs :
            - etfs1 : spectrum
    output : 
            - Ef, th1m ... 

    '''
    d2r=np.pi/180
    grav=9.81
    wn=(2*np.pi*fren)**2/grav
    wavelength=2*np.pi/wn
    Cg=grav/(4*np.pi*fren) 
    dk=2*np.pi*dfreq/Cg
    [nf,nt]=np.shape(efth)
    dir2=np.tile((dirn*d2r).reshape(1,nt),(nf,1))
    Ef=np.sum(efth,             axis=1)*dth
    inds=np.where(wavelength < cut)[0]
    Etot=np.sum(Ef[inds]*dfreq[inds])
    eftn=0.5*(efth+np.roll(efth,nt//2,axis=1))

    #a1E=np.sum(efth*np.cos(dir2),axis=1)*dth
    #b1E=np.sum(efth*np.sin(dir2),axis=1)*dth
    a1=np.zeros(nf)
    b1=np.zeros(nf)
    m1=np.zeros(nf)
    Q1=np.zeros(nf)
    Q2=0
  #  print('TEST:',np.shape(dirn),np.shape(efth),dirn,dth)
    for ind in range(nf):
       #Ef[ind]=np.sum(efth[ind,:]                    )*dth
       if Ef[ind] > 1E-8:
          a1[ind]=np.sum(efth[ind,:]*np.cos(dirn[:]*d2r))*dth/Ef[ind]
          b1[ind]=np.sum(efth[ind,:]*np.sin(dirn[:]*d2r))*dth/Ef[ind]
       else:
          a1[ind]=0.
          b1[ind]=0.
       m1[ind]=np.sqrt(a1[ind]**2+b1[ind]**2)
       Q1[ind]=np.sum(eftn[ind,:]**2)*dth
       Q2=Q2+Q1[ind]*dfreq[ind]*grav**2/(2*((np.pi*2)**4*fren[ind]**3 ))
    #   print('ft:',ind,fren[ind],wn[ind],'Cg:',Cg[ind],dfreq[ind]/(wn[ind]*dk[ind]),grav**2/(2*((np.pi*2)**4*fren[ind]**3 )) )
    Qkk=np.sqrt(Q2)/Etot
    Qf=np.sqrt(np.sum(Ef**2*dfreq))/Etot

    Tm0m1=np.sum(Ef[inds]*dfreq[inds]/fren[inds])/Etot
    Em2=np.sum(Ef[inds]*dfreq[inds]*fren[inds]**2)+Ef[-1]*dfreq[-1]*0.5*fren[-1]**3  # integral including f^-5 tail 
    Tm02=np.sqrt(Etot/Em2)
    Hs=4*np.sqrt(Etot) 
  #     print('WHAT:',ind,fren[ind],Ef[ind],a1[ind],b1[ind],m1[ind],efth[ind,:])
    th1m=np.arctan2(b1,a1)/d2r
  #  print('TEST:',nf,nt,m1,'##',np.sum(efth[5,:]*np.cos(dirn))*dth/Ef[5],a1[5],b1[5],m1[5])
    sth1m=np.sqrt(np.abs(2.0*(1-m1)))/d2r
    return Ef,th1m,sth1m,Hs,Tm0m1,Tm02,Qf,Qkk

#############################################################################
def  wavespec_Ekxky_to_first3(Ekxky,kx2,ky2,f_max=0.15,trackangle=0,depth=1000)  :
    '''
    Computes first 3 moments from E(kx,ky) spectrum
    inputs :
            - Ekxky : spectrum
    output : 
            - Ef, th1m, sth1m ... 

    '''
    ky=ky2[:,0]   # warning: these are in cycles / m, not rad/m 
    kx=kx2[0,:]
    tpi=np.pi*2
    d2r=np.pi/180
    grav=9.81
    k2=np.sqrt(kx2**2+ky2**2)
    f2=np.sqrt((grav*tpi)*k2*np.tanh(k2*(tpi*depth)))/tpi
    Cg2=f2/(2*k2)  # need depth correction
    jacobian=k2/Cg2  
    
    cartesian_interpolator = interp.RegularGridInterpolator((ky, kx), Ekxky*jacobian, method='linear', fill_value=None)
# Define the polar grid (r, theta)
    df=0.005
    f_min = 0.025
    num_f=int((f_max-f_min)/df)+1

    fren = np.linspace(f_min, f_max, num_f)
    dfreq=np.zeros(num_f)+df
    km=(2*np.pi*fren)**2/(grav*2*np.pi)   # cycles / meter
    
    print('Interp:',fren)
    for ii in range(num_f):
       km[ii]=k_from_f(fren[ii],D=depth)/(2*np.pi)            # finite water depth


    nth=72
    theta = np.linspace(0, 360,nth+1)
    dth=(theta[1]-theta[0])*d2r
    K, Theta = np.meshgrid(km, theta*d2r)


    
# Convert polar grid to Cartesian coordinates (x, y)
    X_polar = K * np.cos(Theta-trackangle*d2r)
    Y_polar = K * np.sin(Theta-trackangle*d2r)

    cartesian_points = np.vstack([X_polar.ravel(), Y_polar.ravel()]).T
    polar_values = cartesian_interpolator(cartesian_points)

# Step 6: Reshape the interpolated values back into the polar grid shape
    efth = polar_values.reshape(K.shape).T[:,0:nth]

  
    [nf,nt]=np.shape(efth)
    Ef=np.sum(efth,             axis=1)*dth
    Etot=np.sum(Ef[:]*df)
    eftn=0.5*(efth+np.roll(efth,nt//2,axis=1))

    a1=np.zeros(nf)
    b1=np.zeros(nf)
    m1=np.zeros(nf)
    Q1=np.zeros(nf)
    Q2=0
  #  print('TEST:',np.shape(dirn),np.shape(efth),dirn,dth)
    for ind in range(nf):
       #Ef[ind]=np.sum(efth[ind,:]                    )*dth
       a1[ind]=np.sum(efth[ind,:]*np.cos(theta[0:nth]*d2r))*dth/Ef[ind]
       b1[ind]=np.sum(efth[ind,:]*np.sin(theta[0:nth]*d2r))*dth/Ef[ind]
       m1[ind]=np.sqrt(a1[ind]**2+b1[ind]**2)
       Q1[ind]=np.sum(eftn[ind,0:nth]**2)*dth
       Q2=Q2+Q1[ind]*dfreq[ind]*grav**2/(2*((np.pi*2)**4*fren[ind]**3 ))
    #   print('ft:',ind,fren[ind],wn[ind],'Cg:',Cg[ind],dfreq[ind]/(wn[ind]*dk[ind]),grav**2/(2*((np.pi*2)**4*fren[ind]**3 )) )
    Qkk=np.sqrt(Q2)/Etot
    Qf=np.sqrt(np.sum(Ef**2*dfreq))/Etot

    Tm0m1=np.sum(Ef*dfreq/fren)/Etot
    Em2=np.sum(Ef*dfreq*fren**2)+Ef[-1]*dfreq[-1]*0.5*fren[-1]**3  # integral including f^-5 tail 
    Tm02=np.sqrt(Etot/Em2)
    Hs=4*np.sqrt(Etot) 
  #     print('WHAT:',ind,fren[ind],Ef[ind],a1[ind],b1[ind],m1[ind],efth[ind,:])
    th1m=np.arctan2(b1,a1)/d2r
  #  print('TEST:',nf,nt,m1,'##',np.sum(efth[5,:]*np.cos(dirn))*dth/Ef[5],a1[5],b1[5],m1[5])
    sth1m=np.sqrt(np.abs(2.0*(1-m1)))/d2r

# Checking total energy: 
    Etot=np.sum(efth)*df*dth
    E_mask=np.where( k2 < km[-1], Ekxky, 0) 
    
    
    dkx=kx[1]-kx[0]
    dky=ky[1]-ky[0]
    Etotxy=np.sum(Ekxky*dkx*dky)
    Etotm=np.sum(E_mask*dkx*dky)
    
    print('H:',4*np.sqrt(Etot),4*np.sqrt(Etotxy),4*np.sqrt(Etotm))
    
    return efth,fren,theta[0:nth],Ef,th1m,sth1m


#############################################################################
def  wavespec_Efth_to_Uss(efth,fren,dfreq,dirn,dth)  :
    '''
    Computes first 3 moments from E(f,theta) spectrum
    inputs :
            - etfs1 : spectrum
    output : 
            - Ef, th1m ... 

    '''
    d2r=np.pi/180
    grav=9.81
    sig=(2*np.pi*fren)
    wn=(sig)**2/grav   # warning this is only valid for deep water 
    wavelength=2*np.pi/wn
    Cg=grav/(4*np.pi*fren) 
    dk=2*np.pi*dfreq/Cg
    [nf,nt]=np.shape(efth)
    dir2=np.tile((dirn*d2r).reshape(1,nt),(nf,1))
    a1=np.zeros(nf)
    b1=np.zeros(nf)
    Ef=np.sum(efth,             axis=1)*dth
    Etot=np.sum(Ef*dfreq)
    Hs=4*np.sqrt(Etot) 
    for ind in range(nf):
       a1[ind]=np.sum(efth[ind,:]*np.cos(dirn[:]*d2r))*dth
       b1[ind]=np.sum(efth[ind,:]*np.sin(dirn[:]*d2r))*dth
    Ussx=2*np.sum(a1*wn*sig*dfreq)  # warning this is only valid for deep water 
    Ussy=2*np.sum(b1*wn*sig*dfreq)
    return Hs,Ussx,Ussy

#############################################################################
def  wavespec_Efth_to_first5(efth,fren,dfreq,dirn,dth)  :
    '''
    Computes first 5 moments from E(f,theta) spectrum
    inputs :
            - etfs1 : spectrum
    output : 
            - Ef, th1m ... 

    '''
    d2r=np.pi/180
    grav=9.81
    wn=(2.*np.pi*fren)**2/grav
    Cg=grav/(4*np.pi*fren) 
    dk=2.*np.pi*dfreq/Cg
    [nf,nt]=np.shape(efth)
    dir2=np.tile((dirn*d2r).reshape(1,nt),(nf,1))
    Ef=np.sum(efth,             axis=1)*dth
    Etot=np.sum(Ef*dfreq)
    eftn=0.5*(efth+np.roll(efth,nt//2,axis=1))

    #a1E=np.sum(efth*np.cos(dir2),axis=1)*dth
    #b1E=np.sum(efth*np.sin(dir2),axis=1)*dth
    a1=np.zeros(nf)
    b1=np.zeros(nf)
    m1=np.zeros(nf)
    a2=np.zeros(nf)
    b2=np.zeros(nf)
    m2=np.zeros(nf)
    Q1=np.zeros(nf)
    Q2=0
  #  print('TEST:',np.shape(dirn),np.shape(efth),dirn,dth)
    for ind in range(nf):
       #Ef[ind]=np.sum(efth[ind,:]                    )*dth
       a1[ind]=np.sum(efth[ind,:]*np.cos(dirn[:]*d2r))*dth/Ef[ind]
       b1[ind]=np.sum(efth[ind,:]*np.sin(dirn[:]*d2r))*dth/Ef[ind]
       m1[ind]=np.sqrt(a1[ind]**2+b1[ind]**2)
       a2[ind]=np.sum(efth[ind,:]*np.cos(2*dirn[:]*d2r))*dth/Ef[ind]
       b2[ind]=np.sum(efth[ind,:]*np.sin(2*dirn[:]*d2r))*dth/Ef[ind]
       m2[ind]=np.sqrt(a2[ind]**2+b2[ind]**2)

       Q1[ind]=np.sum(eftn[ind,:]**2)*dth
       Q2=Q2+Q1[ind]*dfreq[ind]*grav**2/(2*((np.pi*2)**4*fren[ind]**3 ))
    #   print('ft:',ind,fren[ind],wn[ind],'Cg:',Cg[ind],dfreq[ind]/(wn[ind]*dk[ind]),grav**2/(2*((np.pi*2)**4*fren[ind]**3 )) )
    Qkk=np.sqrt(Q2)/Etot
    Qf=np.sqrt(np.sum(Ef**2*dfreq))/Etot
    Tm0m1=np.sum(Ef*dfreq/fren)/Etot
    Hs=4*np.sqrt(Etot) 
  #     print('WHAT:',ind,fren[ind],Ef[ind],a1[ind],b1[ind],m1[ind],efth[ind,:])
    th1m=np.arctan2(b1,a1)/d2r
    th2m=np.arctan2(b2,a2)/d2r
  #  print('TEST:',nf,nt,m1,'##',np.sum(efth[5,:]*np.cos(dirn))*dth/Ef[5],a1[5],b1[5],m1[5])
    sth1m=np.sqrt(np.abs(2.0*(1-m1)))/d2r
    sth2m=np.sqrt(np.abs(0.5*(1-m2)))/d2r
    return Ef,th1m,sth1m,th2m,sth2m, Hs,Tm0m1,Qf,Qkk


#############################################################################
def wavespec_MEM(a0,a1,a2,b1,b2, ndirs):
    """(a1,a2,b1,b2,en,ndirs):
% This function calculates the Maximum Entropy Method estimate of
% the Directional Distribution of a wave field.
%
% NOTE: The normalized directional distribution array (NS) and the Energy
% array (NE) have been converted to a geographic coordinate frame in which
% direction is direction from.... but that assumes a1,b1 ... use same convention
%
% First Version: 1.0 - 8/00
%
% Latest Version: 1.0 - 8/00

%
% calculate directional energy spectrum based on Maximum Entropy Method (MEM)
% of Lygre & Krogstad, JPO V16 1986.
%
% switch to Krogstad notation
%   # Maximum entropy method to estimate the Directional Distribution
    
    # Maximum Entropy Method - Lygre & Krogstad (1986 - JPO)
    # Eqn. 13:
    # phi1 = (c1 - c2c1*)/(1 - abs(c1)^2)
    # phi2 = c2 - c1phi1
    # 2piD = (1 - phi1c1* - phi2c2*)/abs(1 - phi1exp(-itheta) -phi2exp(2itheta))^2
    # c1 and c2 are the complex fourier coefficients
    
    """
    nfreq = np.size(a0)
    dr=np.pi/180
    dtheta=360/ndirs
    dirs=np.arange(0.,ndirs,1.)*dtheta
    #print(nfreq,ndirs,np.size(dirs),dirs)
    
    c1 = a1+1j*b1
    c2 = a2+1j*b2
    p1 = (c1-c2*np.conj(c1))/(1.-abs(c1)**2)
    p2 = c2-c1*p1
    
    # numerator(2D) : x
    x = 1.-p1*np.conj(c1)-p2*np.conj(c2)
    x = np.tile(np.real(x),(ndirs,1)).T
    
    # denominator(2D): y
    a = dirs*dr
    e1 = np.tile(np.cos(a)-1j*np.sin(a),(nfreq,1))
    e2 = np.tile(np.cos(2*a)-1j*np.sin(2*a),(nfreq,1))
    
    p1e1 = np.tile(p1,(ndirs,1)).T*e1
    p2e2 = np.tile(p2,(ndirs,1)).T*e2
    
    y = abs(1-p1e1-p2e2)**2
    
    D = x/(y)
    
    # normalizes the spreading function,
    # so that int D(theta,f) dtheta = 1 for each f  
    tot = np.tile(np.sum(D, axis=1),(ndirs,1)).T
    D = D/tot
    
    sp2d = np.tile(a0,(ndirs,1)).T*D/(dr*dtheta)
    
    return sp2d,D,dirs

#############################################################################
###  1. Dispersion relation and associated ##################################

def phase_speed_from_k(k,depth=None,g=9.81):
    if depth == 'None':
        # print("Deep water approximation")
        C=np.sqrt(g/k)
    else:
        # print("General case")
        C=np.sqrt(g*np.tanh(k*depth)/k)
    return C
            
def phase_speed_from_sig_k(sig,k):
    return sig/k
    
def group_speed_from_k(k,depth=None,g=9.81):
    C=phase_speed_from_k(k,depth=depth,g=g)
    if depth == 'None':
        # print("Deep water approximation")
        Cg=C/2
    else:
        # print("General case")
        Cg=C*(0.5+ ((k*depth)/(np.sinh(2*k*depth)) ))
    return Cg

def sig_from_k(k,D=None,g=9.81):
    if D=='None':
        # print("Deep water approximation")
        sig = np.sqrt(g*k)
    else:
        # print("General case")
        sig = np.sqrt(g*k*np.tanh(k*D))
    return sig

def f_from_sig(sig):
    return sig/(2*np.pi)
    
def f_from_k(k,D=None,g=9.81):
    sig = sig_from_k(k,D=D,g=g)
    return sig/(2*np.pi)
    
def sig_from_f(f):
    return 2*np.pi*f
    
def period_from_sig(sig):
    return (2*np.pi)/sig

def period_from_wvl(wvl,D=None):
    k=(2*np.pi)/wvl
    sig=sig_from_k(k,D=D)
    T = period_from_sig(sig)
    return T

def k_from_f(f,D=1000.,g=9.81):
    # inverts the linear dispersion relation (2*pi*f)^2=g*k*tanh(k*dep) to get 
    #k from f and dep. 2 Arguments: f and dep. 
    eps=0.000001
    sig=np.array(2*np.pi*f)
    if D > 1000.:
        # print("Deep water approximation")
        k=sig**2/g
    else:
        Y=D*sig**2/g
        X=np.sqrt(Y)
        I=1
        F=1.
        while (abs(np.max(F)) > eps):
            H=np.tanh(X)
            F=Y-(X*H)
            FD=-H-(X/(np.cosh(X)**2))
            X=X-(F/FD)

        k=X/D

    return k # wavenumber
    
#############################################################################
###  2. Get quick translation info (T0/k/f/L) ###############################
def infos_from_wvl(wvl,D=None):
    wvnb = 2*np.pi/wvl
    f = f_from_k(wvnb,D=D)
    P = 1/f

    print('From a wavelength of ',wvl, ' m : -----------------------')
    print('     - wavenumber k   =   '+f'{wvnb:.4f}'.rjust(6)+' rad/m')
    if D is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ',D,' m')
    print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
    print('     - period T       =   '+f'{P:.2f}'.rjust(6)+' s')
    print('--------------------------------------------------------')

def infos_from_wvnb(wvnb,D=None):
    wvl = 2*np.pi/wvnb
    f = f_from_k(wvnb,D=D)
    P = 1/f

    print('From a wavenumber of ',wvnb, ' rad/m : -----------------------')
    print('     - wavelength L   =   '+f'{wvl:.1f}'.rjust(6)+' m')
    if D is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ',D,' m')
    print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
    print('     - period T       =   '+f'{P:.2f}'.rjust(6)+' s')
    print('--------------------------------------------------------')

def infos_from_T0(P,D=None):
    f = 1/P
    wvnb = k_from_f(f,D=D)
    wvl = 2*np.pi/wvnb

    print('From a period of ',P, ' s : -----------------------')
    print('     - frequency f    =   '+f'{f:.3f}'.rjust(6)+' Hz')
    if D is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ',D,' m')
    print('     - wavelength L   =   '+f'{wvl:.1f}'.rjust(6)+' m')
    print('     - wavenumber k   =   '+f'{wvnb:.4f}'.rjust(6)+' rad/m')
    print('--------------------------------------------------------')

def infos_from_f(f,D=None):
    P = 1/f
    wvnb = k_from_f(f,D=D)
    wvl = 2*np.pi/wvnb

    print('From a frequency of ',f, ' Hz : -----------------------')
    print('     - period T       =   '+f'{P:.2f}'.rjust(6)+' s')
    if D is None:
        print('   With the infinite depth approximation :')
    else:
        print('   With a depth of ',D,' m')
    print('     - wavelength L   =   '+f'{wvl:.1f}'.rjust(6)+' m')
    print('     - wavenumber k   =   '+f'{wvnb:.4f}'.rjust(6)+' rad/m')
    print('--------------------------------------------------------')    



#############################################################################
###  2. Classical Wave spectra ##############################################

## ---- 2.1. 1D Wave spectra along f or k -----------------------------------
def PM_spectrum_f(f,fm,g=9.81):
# There are 2 ways of writing the PM spectrum:
#  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
#  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
# See Hasselmann et al. 1973 for the explanation 
    alpha=8.1*10**-3
    E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
    return E

def PM_spectrum_k(k,fm,D=None,g=9.81):
# There are 2 ways of writing the PM spectrum:
#  - eq 12 of Pierson and Moskowitz (1964) with exp(-0.74 * (f/fw)**-4) where fw=g*U10/(2*pi)
#  - eq of Hasselmann et al. 1973 with exp(-5/4 * (f/ fm)**-4) where fm is the max frequency  ...
# See Hasselmann et al. 1973 for the explanation 
    f=sig_from_k(k,D=D)/(2*np.pi)
    Ef = PM_spectrum_f(f,fm,g=g)
    dfdk = dfdk_from_k(k,D=D)
    
    return Ef*dfdk
    
## ---- 2.2. 2D Wave spectra ------------------------------------------------
def define_spectrum_PM_cos2n(k,th,T0,thetam,n=4,D=None):
    Ek=PM_spectrum_k(k,1/T0,D=D)
    dth=th[1]-th[0]
    Eth=np.cos(th-thetam)**(2*n)
    II=np.where(np.cos(th-thetam) < 0)[0]
    Eth[II]=0
    sth=sum(Eth*dth)
    Ekth=np.broadcast_to(Ek,(len(th),len(k)))*np.broadcast_to(Eth,(len(k),len(th))).T /sth
    return Ekth,k,th

def define_Gaussian_spectrum_kxky(kX,kY,T0,theta_m,sk_theta,sk_k,D=None):
    if (len(kX.shape)==1) & (len(kY.shape)==1):
        kX,kY = np.meshgrid(kX,kY)
    elif (len(kX.shape)==1) | (len(kY.shape)==1):
        print('Error : kX and kY should either be: \n      - both vectors of shapes (nx,) and (ny,) \n  OR  - both matrices of shape (ny,nx)')
        print('/!\ Proceed with caution /!\ kX and kY have been flattened to continue running')
        kX = kX.flatten()
        kY = kY.flatten()
    kp = k_from_f(1/T0,D=D)
    # rotation of the grid => places kX1 along theta = theta_m
    kX1 = kX*np.cos(theta_m)+kY*np.sin(theta_m)
    kY1 = -kX*np.sin(theta_m)+kY*np.cos(theta_m)
    
    Z1_Gaussian =1/(2*np.pi*sk_theta*sk_k)* np.exp( - 0.5*((((kX1-kp)**2)/((sk_k)**2))+kY1**2/sk_theta**2))
    
    return Z1_Gaussian,kX,kY


#############################################################################
###  3. Jacobians and variable changes ######################################

## ---- 3.1 Jacobians -------------------------------------------------------
def dfdk_from_k(k,D=None):
    Cg = group_speed_from_k(k,depth=D,g=9.81)
    return Cg/(2*np.pi)

## ----- 3.2 Change variables from spectrum ---------------------------------
def spectrum_from_fth_to_kth(Efth,f,th,D=10000.):
    shEfth = np.shape(Efth)
    # print(shEfth)    
    if len(shEfth)<2:
        print('Error: spectra should be 2D')
    else:
        if shEfth[0]==shEfth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,th)')
        elif (shEfth[1]==len(f)) &(shEfth[0]==len(th)):
            Efth = np.swapaxes(Efth,0,1)
        elif (shEfth[1]==len(th)) &(shEfth[0]==len(f)):
            print('All good: Efth have the shape : (f,th)')    
        else:
            print('shEfth[1] : ',shEfth[1], ' vs ',len(f),'// shEfth[0] :',shEfth[0],' vs ',len(th))
            print('Error: Efth should have the shape : (f,th)')
    shEfth = np.shape(np.moveaxis(Efth,0,-1))
    k=k_from_f(f,D=D)
    dfdk=dfdk_from_k(k,D=D)
    Ekth = Efth*np.moveaxis(np.broadcast_to(dfdk,shEfth),-1,0)
    

    return Ekth, k, th

def spectrum_from_kth_to_kxky(Ekth,k,th):
    try:
        shEkth = np.shape(Ekth)
        # print(shEkth,np.shape(k),np.shape(th))  
        if len(shEkth)<2:
            print('Error: spectra should be 2D')
        else:
            if shEkth[0]==shEkth[1]:
                print('Warning: same dimension for k and theta.\n  Proceed with caution: The computation is done considering Ekth = f(k,th)')
            elif ((shEkth[1]==len(k)) &(shEkth[0]==len(th))) | ((shEkth[1]==len(th)) &(shEkth[0]==len(k))):
                if (shEkth[1]==len(k)) &(shEkth[0]==len(th)):
                    Ekth = np.swapaxes(Ekth,0,1)
            else:
                print('shEkth[1] : ',shEkth[1], ' vs ',len(k),'// shEkth[0] :',shEkth[0],' vs ',len(th))
                print('Error: Ekth should have the shape : (k,th)')
        shEkth2 = np.shape(np.moveaxis(Ekth,0,-1)) # send k-axis to last -> in order to broadcast k along every dim
        shEkth2Dkth = Ekth.shape[0:2] # get only shape k,th for the broadcast of the dimensions kx,ky
        # print('shEkth2Dkth : ',shEkth2Dkth)
        if np.max(th)>100:
            th=th*np.pi/180
        kx = np.moveaxis(np.broadcast_to(k,shEkth2Dkth[::-1]),-1,0) * np.cos(np.broadcast_to(th,shEkth2Dkth))
        ky = np.moveaxis(np.broadcast_to(k,shEkth2Dkth[::-1]),-1,0) * np.sin(np.broadcast_to(th,shEkth2Dkth))
        Ekxky = Ekth/np.moveaxis(np.broadcast_to(k,shEkth2),-1,0)
        #print(np.shape(Ekxky))
        return Ekxky, kx, ky
    except Exception as inst:
        print('in spec to kxky : ',inst,', line number : ',sys.exc_info()[2].tb_lineno)

def spectrum_from_fth_to_kxky(Efth,f,th,D=10000.):
    shEfth = np.shape(Efth)
    #print(shEfth)
    if len(shEfth)<2:
        print('Error: spectra should be 2D')
    else:
        if shEfth[0]==shEfth[1]:
            print('Warning: same dimension for freq and theta.\n  Proceed with caution: The computation is done considering Efth = f(f,th)')
        elif ((shEfth[1]==len(f)) &(shEfth[0]==len(th))) | ((shEfth[1]==len(th)) &(shEfth[0]==len(f))):
            if (shEfth[1]==len(f)) &(shEfth[0]==len(th)):              
                Efth = np.swapaxes(Efth,0,1)  
        else:
            print('Error: Efth should have the shape : (f,th)')
    shEfth2 = np.shape(np.moveaxis(Efth,0,-1)) # send f-axis to last -> in order to broadcast f along every dim
    shEfth2Dfth = Efth.shape[0:2] # get only shape f,th for the broadcast of the dimensions kx,ky
    k=k_from_f(f,D=D)
    dfdk=dfdk_from_k(k,D=D)
    if np.max(th)>100:
        th=th*np.pi/180

    kx = np.moveaxis(np.broadcast_to(k,shEfth2Dfth[::-1]),-1,0) * np.cos(np.broadcast_to(th,shEfth2Dfth))
    ky = np.moveaxis(np.broadcast_to(k,shEfth2Dfth[::-1]),-1,0) * np.sin(np.broadcast_to(th,shEfth2Dfth))
    Ekxky = Efth * np.moveaxis(np.broadcast_to(dfdk /k,shEfth2),-1,0)
    return Ekxky, kx, ky

def spectrum_to_kxky(typeSpec,Spec,ax1,ax2,D=None):
    if typeSpec==0: # from f,th
        Ekxky, kx, ky = spectrum_from_fth_to_kxky(Spec,ax1,ax2,D=D)
    elif typeSpec==1: # from k,th
        Ekxky, kx, ky = spectrum_from_kth_to_kxky(Spec,ax1,ax2)
    else:
        print('Error ! typeSpec should be 0 = (f,th) or 1 = (k,th)')
        Ekxky = Spec
        kx = ax1
        ky = ax2
    return Ekxky, kx, ky

def spectrum_f_to_k(Ef,f,D=None):
    shEf = np.array(np.shape(Ef))
    ind = np.where(shEf == len(f))[0]
    if len(ind)==0:
        print('Error: spectra should have an axis with same dimension as f')
    elif len(ind)>1:
        print('Warning: same dimension for different axes.\n  Proceed with caution: The computation is done considering Ef = f(...,f)')
        if ind[-1]<(len(shEf)-1):
            Ef=np.swapaxes(Ef,ind[-1],-1)
    elif len(ind)==1:
        Ef=np.swapaxes(Ef,ind,-1) # pass the f axis as last dim : to broadcast 
    
    k=k_from_f(f,D=D)
    dfdk=dfdk_from_k(k,D=D)
    shEf2 = np.shape(Ef)
    Ek = np.swapaxes(Ef*np.broadcast_to(dfdk,shEf2),-1,ind)
    return Ek, k

def spectrum_k_to_f(Ek,k,D=None):
    shEk = np.array(np.shape(Ek))
    ind = np.where(shEk == len(k))[0]
    if len(ind)==0:
        print('Error: spectra should have an axis with same dimension as k')
    elif len(ind)>1:
        print('Warning: same dimension for different axes.\n  Proceed with caution: The computation is done considering Ek = f(...,k)')
        if ind[-1]<(len(shEk)-1):
            ind0 = int(ind[-1])
            Ek=np.swapaxes(Ek,ind0,-1)
    elif len(ind)==1:
        ind0 = int(ind)
        Ek=np.swapaxes(Ek,ind0,-1) # pass the f axis as last dim : to broadcast 
    
    f=f_from_k(k,D=D)
    dfdk=dfdk_from_k(k,D=D)
    shEk2 = np.shape(Ek)
    Ef = np.swapaxes(Ek/np.broadcast_to(dfdk,shEk2),-1,ind0)
    return Ef, f
    
#############################################################################
def PM_spectrum_k(k,fm,g=9.81):
    #pmofk(k,T0,H)
    alpha=8.1*10**-3
    
    w0=2*np.pi/T0
    w=np.sqrt(g*k*tanh(k*H))
    #Cg=(0.5+k.*H/sinh(2.*k.*H)).*w./k;
    #pmofk=0.008.*g.^2.*exp(-0.74.*(w./w0).^(-4))./(w.^5).*Cg+5.9;
    
    E = alpha*g**2*(2*np.pi)**-4*f**-5*np.exp((-5/4)*((fm/f)**4))
    return E
#############################################################################

