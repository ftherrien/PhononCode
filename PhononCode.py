import numpy as np
import matplotlib
from numpy import linalg as la
import scipy.fftpack
import time
import random as rd
import os, os.path
import argparse
from mpi4py import MPI
import sys
import pickle
import re

def cell(s):
    if (s[0]=="[") and (s[-1]=="]"):
        try:
            s=s[1:-1]
            tup = map(float, s.split(','))
            return list(tup)
        except:
            raise argparse.ArgumentTypeError("Should be [x,y,z,...] of size nb")
    else:
        raise argparse.ArgumentTypeError("Should be [x,y,z,...] of size nb")

def bonds(s):
    if (s[0]=="[") and (s[-1]=="]"):
        try:
            s=s[1:-1]
            dTypeList = re.findall('\[[0-9,\,,\., ]*\]|(?<=\,)[^,]*(?=\,)|(?<=\,)[^,]*$',s)
            NonList=0
            InitLen = len(dTypeList)
            for i,s in enumerate(list(dTypeList)):
                if (s=='' or  s==' ') and i==InitLen-1 and NonList==1:
                    dTypeList[i]=[]
                else:
                    if (s[0]=="[") and (s[-1]=="]"):
                        s=s[1:-1]
                        if s=='' or  s==' ':
                            dTypeList[i]=[]
                        else:
                            dTypeList[i] = list(map(float, s.split(',')))
                    else:
                        NonList=+1
                        if NonList==1:
                            dTypeList[i]=[float(s),float(s)]
                            if i == InitLen-1:
                                dTypeList.append([])
                        else:
                            dTypeList[i]=[float(s)]
                    
            return [np.array(dTypeList[:-1]),dTypeList[-1]]
        except:
            raise argparse.ArgumentTypeError("Should be of form [e_i,[e_{0l},e_{0r}],...,[e_{(nd-1)l},e_{(nd-1)r}],[i_1,i_2,...,i_{nb-1}]]")
    else:
        raise argparse.ArgumentTypeError("Should be a python list []")

parser = argparse.ArgumentParser()
parser.add_argument("-b","--pcsize",dest="nb",type=int, default=1, help="Size of primitive cell")
parser.add_argument("-n","--scsize",dest="n", type=int, default=50, help="Number of primitive cell in super cell")
parser.add_argument("-E","--energy",dest="nE", type=int, default=600, help="Number of energy values on scale")
parser.add_argument("-T","--temp",dest="T", type=float, default=300, help="Temperature")
parser.add_argument("-V","--potentials",dest="V", type=cell, default=[1], help="List of potential strenghts of size nb")
parser.add_argument("-M","--masses",dest="Mvec", type=cell, default=[1], help="List of masses of size nb")
parser.add_argument("-g","--noGauss",dest="gaussian", action="store_false", default=True, help="Energy bands with minimum gaussian width")
parser.add_argument("-u","--cutOffErr",dest="CutOffErr", type=float, default=10**-4, help="Value at witch a band is considered to be negligable")
parser.add_argument("-w","--width",dest="w", default=-1, type=float, help="Standard deviation (width) of gaussian")
parser.add_argument("-d","--defects",dest="defects", action="store_true", default=False, help="Adds defects in super cell")
parser.add_argument("-t","--dtype",dest="dtype",nargs='+', type=str, default=["random"], help="Available types: ordered, random, cluster")
parser.add_argument("-s","--cluster",dest="clusterSize",nargs='+',type=float, default=[3], help="Size factor for clustered defects")
parser.add_argument("-m","--dmasses",dest="mval",nargs='+',type=cell, default=[[2]], help="Mass of defects")
parser.add_argument("-v","--dpotentials",dest="kval",nargs='+',type=bonds, default=[[2]], help="Potential strenghts of defects")
parser.add_argument("-c","--defConc",dest="DefConc",nargs='+',type=float, default=[0.05], help="Concentration of defect")
parser.add_argument("-o","--out",dest="folder",type=str, default="images", help="Output folder for images")
parser.add_argument("-i","--display",dest="disp",action="store_true", default=False, help="Display generated plots plots")

options = parser.parse_args()

# Initial calls
comm = MPI.COMM_WORLD
master = 0
n_proc = comm.Get_size()
rank = comm.Get_rank()

# Display initialisation

if options.disp:
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ioff
else:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


## PARAMETERS #######################################################################
c=1 #Interatomic spacing
nb = options.nb #Number of particles per primitive cell
n = options.n #Initial number of primitive cell in super cell
nE = options.nE #Number of energy values on scale
Na = 1 #Total number of SC in B-vKC
Nb = n #Total number of PC in B-vKC
Nc = n*nb #Total number of atoms in Born-von Karman cell
T = options.T #Temperature in Kelvin

# Potential
V=np.array(options.V)*(1+0*1j) #Potential vector (without defects)
# Masses
Mvec = np.array(options.Mvec) # Mass vector (without defects)

gaussian = options.gaussian # Gaussian minimal width
CutOffErr=options.CutOffErr # Cuttoff value for energy difference
if options.w == -1:
    w = 1/n # width of gaussian in fraction of the max energy
else:
    w = options.w
# Good width: 0.035

defects = options.defects
#Presence of defects
# dtypes:
# ordered: will repeat defect periodically to obtain DefConc, if multiple defect types are specified they will be
# stacked next to each other and will have the same concentration other concentrations will be ignored
# random: defects are scattered randomly in the supercell, different types of defects can have different concentrations
# cluster: defects are have higher probability to be near another defect, the location of the first defect is random. The clusterSize variable controls the standanrd deviation of the density of probability
dtype = options.dtype
clusterSize = options.clusterSize
mval = np.array(options.mval)
kval = options.kval
for im,m in enumerate(mval):
    if len(kval[im][-1])+1 != len(m):
        raise argparse.ArgumentTypeError("The defect mass and potential should be the same size")

if defects:
    DefConc = options.DefConc #concentratation of defects
else:
    DefConc = [0]

#Images output folder
folder = options.folder+"/"
namestamp = "phon"

#Constant
hbar_kb=7.63824*10**(-12) #second kelvins

## PARAMETERS #######################################################################
if rank==master:
    if (not os.path.exists(folder)):
        os.mkdir(folder)

    # Writting PARAM.out
    f = open(folder+"PARAM.out", 'w')
    print(options,file=f)
    f.close()

    # Timing
    t_total_i = time.time()

#Errors -----------------------------------------------------------------------------

MacPrecErr= 2*np.finfo(float).eps
GaussErr = 1.1**3 / (12*np.sqrt(np.pi) * w**3 * nE**2)

#Primitive Cell ---------------------------------------------------------------------
b=nb*c #Size of lattice
M = np.diag(Mvec, 0)

#Super Cell -------------------------------------------------------------------------
na=n*nb#Number of particles per cell
a = na*c #Size of lattice

#Potential
Vsc=[np.array(V) for i in range(n)] #np.tile(V,(n,1))
#Masses
Mvecsc=[np.array(Mvec) for i in range(n)] #np.tile(Mvec,(n,1))

def load_balance(n_tasks):
# Defines the interval each cores needs to compute
    
    n_jobs = n_tasks//n_proc
    balance = n_tasks%n_proc

    if (rank < balance): 
        i_init = rank*(n_jobs+1)+1
        i_fin = (rank+1)*(n_jobs+1)
    else:
        i_init = balance*(n_jobs+1) + (rank-balance)*(n_jobs)+1
        i_fin = balance*(n_jobs+1) + (rank-balance+1)*(n_jobs)

    return range(i_init-1, i_fin)

def invCumulFunc(folder,namestamp,graph,n,occpos,mu,var,x):
    print(occpos,mu,var,x)
    CumulFunc = np.zeros(n)
    DensProb = np.zeros(n)
    mu = np.append(np.append(mu - n,mu),mu+n) #periodicity considering effect of n+ is negligable
    if (occpos[0] == 0):
        DensProb[0] = np.sum(1 / np.pi * np.exp(-(mu) ** 2 / var))
        CumulFunc[0] = DensProb[0]
    for i in range(1,n):
        if (occpos[i] == 0):
            DensProb[i] = np.sum(1 / np.pi * np.exp(-(i - mu) ** 2 / var))
        CumulFunc[i] = CumulFunc[i - 1] + DensProb[i]
    DensProb = DensProb / CumulFunc[-1]
    CumulFunc = CumulFunc / CumulFunc[-1]
    pos = np.arange(n)[CumulFunc>=x][0]

    if graph:
        plt.figure()
        ax = plt.gca()
        oridens = [np.sum(1 / np.pi * np.exp(-(i - mu) ** 2 / var)) for i in range(n)]
        oridens = oridens / sum(oridens)
        ax.bar(np.arange(n), oridens, color='r', alpha=0.3)
        ax.bar(np.arange(n), DensProb, alpha=0.3)
        plt.ylabel('Density of probability')
        plt.xlabel('Position in supercell')
        plt.title("Density of probability for clustered defects")
        plt.savefig(folder+"defects_"+namestamp+".png")

    return pos

def gcd(x, y):
   """This function implements the Euclidian algorithm
   to find G.C.D. of two numbers"""

   while(y):
       x, y = y, x % y

   return x

# define lcm function
def lcm(x, y):
   """This function takes two
   integers and returns the L.C.M."""

   lcm = (x*y)//gcd(x,y)
   return lcm

if rank == master:
    # Defects
    if defects:
        occpos = np.zeros(n,np.int)

        for i in range(len(mval)):
            
            availpos = list(np.arange(n)[occpos==0])
            ndefects = int(DefConc[i] * n)
            if ndefects == 0:
                print('WARNING: The specified defect concentration is too low for the size of the super cell: there is no defect of type %d'%(i+1),file=sys.stderr)
            else:
                if dtype[i] == "ordered":
                    pos = np.arange(i, n, int(1 / DefConc[i]),np.int)
                    occpos[pos] = i+1

                if dtype[i] == "random":
                    pos = rd.sample(availpos,ndefects)
                    pos = np.array(pos)
                    occpos[pos] = i+1

                if dtype[i] == "cluster":
                    mu = np.array([],np.int)
                    graph = False
                    for k in range(ndefects):

                        if k==0:
                            posk = rd.sample(availpos,1)[0]
                        else:
                            if k == ndefects-1:
                                graph = True
                            posk = invCumulFunc(folder,namestamp,graph,n,occpos,mu,clusterSize[i]**2,rd.random())
                        mu = np.append(mu,posk)
                        occpos[posk] = i+1
                    pos = mu

                nextp = pos + 1
                nextp[nextp==n]=0
                prevp = pos - 1
                prevp[prevp==-1]=n-1
                
                for ip,p in enumerate(pos):
                    Vsc[p-1][-1] = kval[i][0][occpos[prevp[ip]],0]
                    Vsc[p] = np.append(kval[i][1],kval[i][0][occpos[nextp[ip]],1])
                    Mvecsc[p]= mval[i]

    Vsc = np.concatenate(Vsc)
    Mvecsc = np.concatenate(Mvecsc)

    print(Mvecsc)
    print(np.real(Vsc))
    print('---')
    print(occpos)

    # New SuperCell size
    na = len(Mvecsc)
    a = na*c
    Nc = lcm(na,nb)
    Na = Nc//na
    Nb = Nc//nb 

    print(na,nb,Nc,Na,Nb)
    
    # Temporary non optimal fix TODO
    Vsc = np.tile(Vsc,Na)
    Mvecsc = np.tile(Mvecsc,Na)
    na = int(na*Na)
    print("WARNING: to match the PC and the defects, the SC has to be %d times larger"%Na)
    Na = 1
    a = na*c
    
    # Solving for supercell at Q=0 -----------------------------------------------------------------------------------------

    Msc = np.diag(Mvecsc, 0)

    # D matrix
    secdiag = Vsc[0:-1]/np.sqrt(Mvecsc[0:-1]*Mvecsc[1:])
    maindiag = np.hstack((-(Vsc[0] + Vsc[-1])/Mvecsc[0], -(Vsc[1:] + Vsc[0:-1])/Mvecsc[1:]))
    D = (np.diag(secdiag, -1) + np.diag(maindiag, 0) + np.diag(secdiag, 1))
    D[-1, 0] = D[-1, 0] + Vsc[-1]/np.sqrt(Mvecsc[0]*Mvecsc[-1])
    D[0, -1] = D[0, -1] + Vsc[-1]/np.sqrt(Mvecsc[0]*Mvecsc[-1])

    #print(K)

    # System -w^2*x=Dx
    sysmat = D

    # Solving system
    omegasqsc, eigenvecsc = la.eigh(-sysmat)

    omegasqsc=np.real(omegasqsc)

    idx = omegasqsc.argsort()
    omegasqsc = omegasqsc[idx]
    omegasqsc[omegasqsc<0]=0
    omegasc = np.sqrt(omegasqsc)
    eigenvecsc = eigenvecsc[:,idx]

    
    # Gram-Schmidt Process (modified considering most vectors are orthonormal)
    eigenvecN = np.array(eigenvecsc)
    for i in range(na):
        for j in np.arange(i+1,na):
            if (abs(omegasc[i] - omegasc[j]) < MacPrecErr):
                eigenvecN[:, i] = eigenvecN[:, i] - eigenvecsc[:, i].dot(np.conj(eigenvecN[:, j])) * eigenvecN[:, j]
        if ((eigenvecN[:, i] != eigenvecsc[:, i]).all):
            eigenvecN[:, i] = eigenvecN[:, i] / la.norm(eigenvecN[:, i])

    eigenvecsc = np.array(eigenvecN)
else:
    eigenvecsc = None
    omegasc = None
    

eigenvecsc = comm.bcast(eigenvecsc)
omegasc = comm.bcast(omegasc)

tol = omegasc.max()*1.1/(2*(nE-1)) #Tolerence for equality

# Solving for primitive cell at q = Q + G ------------------------------------------------------------------------------

q=np.arange(0,1/2+1/(Nb),1/(Nb))*(2*np.pi/b)

dE = omegasc.max()*1.1/(nE-1)
sig = w*omegasc.max()
if gaussian:
    Emin = - sig * np.sqrt( np.log( dE / ( CutOffErr * sig * np.sqrt( np.pi ) ) ) )
    nE = nE+int(np.ceil(-Emin/dE))
    Emin = np.ceil(Emin/dE)*dE
else:
    Emin = 0
E=np.linspace(Emin,omegasc.max()*1.1,nE)

Local_range = load_balance(len(q))

Local_Sf = np.zeros((nb,nE,len(Local_range)))
Local_Sftotal = np.zeros((nE,len(Local_range)))
Local_omegadisp=np.zeros((nb,len(Local_range)))

Local_qLoopTimes = []

t_total_q = time.time()

for jq,iq in enumerate(Local_range): # Wave vector times lattice vector (1D) [-pi, pi]

    #timing
    t_q_i = time.time()

    #D matrix
    secdiag = V[0:-1]/np.sqrt(Mvec[0:-1]*Mvec[1:])
    maindiag = np.hstack((-(V[0] + V[-1])/Mvec[0], -(V[1:] + V[0:-1])/Mvec[1:]))
    D = (np.diag(secdiag, -1) + np.diag(maindiag, 0) + np.diag(secdiag, 1))
    D[-1, 0] = D[-1, 0] + V[-1]*np.exp(-1j*q[iq]*b)/np.sqrt(Mvec[0]*Mvec[-1])
    D[0, -1] = D[0, -1] + V[-1]*np.exp( 1j*q[iq]*b)/np.sqrt(Mvec[0]*Mvec[-1])

    #System -w^2*x=Dx
    sysmat = D

    #Solving system
    omegasq, eigenvec = la.eigh(-sysmat)

    #Order the bands
    idx = omegasq.argsort()
    omegasq = omegasq[idx]
    eigenvec = eigenvec[:,idx]

    omegasq[omegasq < 0] = 0
    omega=np.sqrt(omegasq)

    omega=np.real(omega)

    #Gram-Schmidt Process (modified considering most vectors are orthonormal)
    saveeigen = np.array(eigenvec)
    eigenvecN = np.array(eigenvec)
    for i in range(nb):
        for j in np.arange(i+1,nb):
            if (abs(omega[i]-omega[j])<MacPrecErr):
                eigenvecN[:,i]=eigenvecN[:,i]-eigenvec[:,i].dot(np.conj(eigenvecN[:,j]))*eigenvecN[:,j]
        if ((eigenvecN[:,i] != eigenvec[:,i]).all):
            eigenvecN[:, i]=eigenvecN[:,i]/la.norm(eigenvecN[:,i])

    eigenvec=np.array(eigenvecN)

    # Original band structure
    Local_omegadisp[:,jq]=omega 
    
    #t_energy_loop = time.time()
    #t_i_loop = np.zeros(nE)
    deltalist=np.zeros(len(E))
    for iE in range(nE):

        # timing
        #t_i_i = time.time()
        #t_l_loop = np.zeros(na)

        ############[ SUM ON ALL STATES ]############

        # sum i=1->na, outermost sum in definition of Sf
        for i in range(na):

            ############[ DIRAC DELTA FUNCTION ]############

            # delta (E - epsi)
            # print(omegasc[i])
            # print(dE)
            # print(delta)
            if gaussian == True:
                delta = dE / (sig*np.sqrt(np.pi)) * np.exp(-(omegasc[i] - E[iE]) ** 2 / sig**2)
                condition = (delta > CutOffErr)
            else:
                condition = (abs(omegasc[i] - E[iE]) < tol)
                delta = 1
            if condition:

                deltalist[iE] = deltalist[iE] + delta

                ScalarProd = np.zeros(nb)*(1+0*1j)
#                count[iq,i] = count[iq,i] + 1
                #timing
                #t_l_i = time.time()

                ############[ SCALAR PRODUCT ]############

                # sum l=1->Nt on all space for the scalar product
                for l in range(Nc):
                    # sum s=1->nb on all solutions of the primitive cell
                    for s in range(nb):
                        ScalarProd[s] = ScalarProd[s] + 1/np.sqrt(Na*Nb)*np.conj(eigenvecsc[l%na,i])*eigenvec[l%nb,s]*np.exp(-1j*q[iq]*(l//nb)*b)

                # ////////////[ SCALAR PRODUCT ]////////////
                for s in range(nb):
                    Local_Sf[s, iE, jq] = Local_Sf[s, iE, jq] + delta * abs(ScalarProd[s]) ** 2
                Local_Sftotal[iE, jq] = Local_Sftotal[iE, jq] + delta * abs(np.sum(ScalarProd)) ** 2

                #timing
                #t_l_loop[i] = t_l_loop[i] + time.time()-t_l_i

            #////////////[ DIRAC DELTA FUNCTION ]////////////

        # ////////////[ SUM ON ALL STATES ]////////////

        #timing
        #t_i_loop[iE] = t_i_loop[iE]+time.time()-t_i_i
#        print('Total times for Loop in l=', t_l_loop)

    #timing
    #print('Total times for Loop in i=', t_i_loop)
    t_q_f = time.time()
    Local_qLoopTimes.append(t_q_f - t_q_i)
    print("Finished %d in core %d in %f seconds"%(iq,rank,t_q_f - t_q_i))
    #print('Energy loop time=',t_q_f-t_energy_loop)
    #print('System solve + GS proces=', -t_q_i + t_energy_loop)

    #print('-------------------------------------------------')

Global_omegadisp = comm.gather(Local_omegadisp)
Global_Sf = comm.gather(Local_Sf)
Global_Sftotal = comm.gather(Local_Sftotal)
qLoopTimes = comm.gather(Local_qLoopTimes)

if rank == master:

    omegadisp = np.concatenate(Global_omegadisp,1)
    Sf = np.concatenate(Global_Sf,2)
    Sftotal = np.concatenate(Global_Sftotal,1)

    # Save resulting matrix
    pickle.dump( Sf, open( folder+"Sf.dat", "wb" ) )
    pickle.dump( Sftotal, open( folder+"Sftotal.dat", "wb" ) )
    
    plt.figure()
    plt.plot(q,omegadisp.T,'.')
    plt.ylabel(r'Angular Frequency($\omega$)')
    plt.xlabel('Wave vector(q)')
    plt.title("Primitive cell band structure")

    plt.savefig(folder+"primitive_band_"+namestamp+".png")

    Sf[Sf<MacPrecErr]=0

    plt.figure()
    plt.imshow(np.sum(Sf,0), interpolation='None', origin='lower',
                    cmap=plt.cm.spectral_r,aspect='auto',extent=[q.min(), q.max(), E.min(), E.max()],vmax=1, vmin=0)
    plt.ylabel(r'Angular Frequency($\omega$)')
    plt.xlabel('Wave vector(q)')
    plt.title("Total band")
    plt.savefig(folder+"spectral_map_"+namestamp+".png")

    plt.figure()
    plt.imshow(Sftotal-np.sum(Sf,0), interpolation='None', origin='lower',
                    cmap=plt.cm.spectral_r,aspect='auto',extent=[q.min(), q.max(), E.min(), E.max()],vmax=1, vmin=0)
    plt.ylabel(r'Angular Frequency($\omega$)')
    plt.xlabel('Wave vector(q)')
    plt.title("Cross terms")
    plt.savefig(folder+"cross_spectral_map_"+namestamp+".png")

    for s in range(nb):
        plt.figure()
        #plt.imshow(Sf, aspect=1, interpolation='none', cmap=plt.get_cmap('Greys'),
        #                origin='lower', extent=[q.min(), q.max(), E.min(), E.max()],
        #                vmax=1, vmin=0)
        # inter: lanczos
        plt.imshow(Sf[s,:,:], interpolation='None', origin='lower',
                        cmap=plt.cm.spectral_r,aspect='auto',extent=[q.min(), q.max(), E.min(), E.max()],vmax=1, vmin=0)
        plt.ylabel(r'Angular Frequency($\omega$)')
        plt.xlabel('Wave vector (q)')
        plt.title("Band %d"%s)
        plt.savefig(folder+"band_%d_spectral_map_"%s+namestamp+".png")
    print('Primitive Cell Frenquecies')
    print(omegadisp)
    print('Super Cell Frequencies')
    print(omegasc)
    print('Validation')
    print('Total')
    print(np.sum(Sftotal,0))
    print('Crossterms')
    print(np.sum(Sftotal-np.sum(Sf,0),0))
    for s in range(nb):
        print('Band %d'%s)
        print(np.sum(Sf[s,:,:],0))

    print('Max Error')
    MaxErr = np.max(abs(nb-np.sum(Sftotal,0)))
    print(MaxErr)
    print('Gaussian Error estimation')
    print(GaussErr)

    #Display
    plt.figure()
    plt.plot(E, Sf[:,:, 0].T, '.-')
    plt.ylabel('Spectral function (a.u.)')
    plt.xlabel(r'Angular Frequency($\omega$)')
    plt.title("Slice of the spectral map at q=0")
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    deltalist = deltalist / max(deltalist)
    for npE in range(len(E)):
        ax.vlines(x=E[npE], ymin=ymin, ymax=ymax, color='r', alpha=deltalist[npE])
    plt.ylim([ymin, ymax])
    plt.plot(E,deltalist*ymax, color='r')

    plt.savefig(folder+"slice_0_"+namestamp+".png")

    #lifetime calculations -------------------------------------------------------------------------------------------------

    EAvg = np.zeros((nb, len(q)))
    EsqAvg = np.zeros((nb, len(q)))
    Var = np.zeros((nb, len(q)))


    for s in range(nb):
        for iE in range(nE):
            EAvg[s,:] = EAvg[s,:] + E[iE]*Sf[s,iE,:]
            EsqAvg[s,:] = EsqAvg[s,:] + (E[iE]**2)*Sf[s,iE,:]
        EAvg[s,:] = EAvg[s,:]/np.sum(Sf[s, :, :], 0)
        EsqAvg[s,:] = EsqAvg[s,:]/np.sum(Sf[s, :, :], 0)

    for s in range(nb):
        for iE in range(nE):
            Var[s,:] = Var[s,:] + (E[iE]-EAvg[s,:])**2*Sf[s,iE,:]
        Var[s,:] = Var[s,:]/np.sum(Sf[s, :, :], 0)

    Var2 = EsqAvg-EAvg**2
    DeltaE = np.sqrt(Var)
    Tau = 1/DeltaE

    # Display
    plt.figure()
    plt.plot(q,EAvg.T)
    for s in range(nb):
        plt.fill_between(q, EAvg[s, :] - DeltaE[s, :], EAvg[s, :] + DeltaE[s, :],color='#6E6E6E',alpha=0.2)
    plt.plot(q,omegadisp.T,'--')
    plt.ylabel(r'Angular Frequency($\omega$)')
    plt.xlabel('Wave vector (q)')
    plt.title("Average frequency and standard deviation")
    plt.savefig(folder+"avg_std_"+namestamp+".png")

    plt.figure()
    plt.plot(q,Tau.T)
    plt.ylabel(r'Lifetime($\tau$)')
    plt.xlabel('Wave vector (q)')
    plt.title("Lifetime")
    plt.savefig(folder+"lifetime_"+namestamp+".png")

    # Lattice Thermal Conductivity -----------------------------------------------------------------------------------------

    dq = 2*np.pi/a
    Vol = b/(2*np.pi)
    bar = hbar_kb*EAvg/T
    C=np.ones((nb,len(q)))
    C[bar>=MacPrecErr] = bar[bar>=MacPrecErr]**2*np.exp(bar[bar>=MacPrecErr])/(np.exp(bar[bar>=MacPrecErr])-1)**2
    v = np.zeros((nb,len(q)))
    v[:,1:-1] = 1/dq*1/2*(EAvg[:,2:]-EAvg[:,:-2])
    v[:,0] = 1/dq*(2*EAvg[:,1]-3/2*EAvg[:,0]-1/2*EAvg[:,2])
    v[:,-1] = 1/dq*(-2*EAvg[:,-2]+3/2*EAvg[:,-1]+1/2*EAvg[:,-3])

    k = 1/Vol*np.sum(C*v**2*Tau,1)
    LTC = np.sum(k)

    print("##################################################")
    print("Total Lattice Thermal Conductivity:", LTC)
    for s in range(nb):
        print("Band %d contribution:"%s, k[s])
    print("--------------------------------------------------")
    print("Prameters:")
    print("Specific Heat (C):", C)
    print("Group velocity (v):", v)
    print("Relaxation Time (Tau):", Tau)
    print("##################################################")

    # Writting to file
    f = open('PHONON.out', 'a')
    printdtype = ' '.join([str(i) for i in dtype])
    printclusterSize = ' '.join([str(i) for i in clusterSize])
    printdefconc = ' '.join([str(i) for i in DefConc])
    printmass = '-'.join([str(i) for i in Mvec])
    printpot = '-'.join([str(i) for i in np.real(V)])
    printdpot = ' '.join(['-'.join([str(j) for j in i]) for i in kval])
    printdmass = ' '.join(['-'.join([str(j) for j in i]) for i in mval])
    print(printmass,printpot,printdefconc,printdtype,printclusterSize,printdpot,printdmass,LTC, file=f)
    f.close

    t_total_f = time.time()

    print('done')
    print('Sequential start time', t_total_f-t_q_f)
    print('Sequential end time',t_total_q - t_total_i)
    print('Total elapsed time:', t_total_f-t_total_i)

    if options.disp:
        plt.show()
