import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy.fftpack
import time
import random as rd

plt.close("all")

# Initial parameters:
c=1 #Interatomic spacing
nb=1 #Number of particles per cell
n=100 #number of primitive cell in super cell
nE = 600 #Number of energy values on scale
Nc=1 #Number of Sc in Von Karman cell
T = 300 #Temperature in Kelvin

gaussian = True
CutOffErr=10**-4 # Cuttoff value for energy difference
w =  1/n#width of gaussian in fraction of the max energy
# Good width: 0.035

defects = False
# dtypes:
# ordered: will repeat defect periodically to obtain DefConc, if multiple defect types are specified they will be
# stacked next to each other and will have the same concentration other concentrations will be ignored
# random: defects are scattered randomly in the supercell, different types of defects can have different concentrations
dtype = ["cluster","random"]
clusterSize = [3,3]
mval = np.array([[2],[3]])
kval = np.array([[2],[3]])
DefConc = [0.05,0.1] #concentratation of defects

#Images output folder
folder = "images\\"

#Constant
hbar_kb=7.63824*10**(-12) #second kelvins

#Errors ----------------------------------------------------------------------------------------------------------------

MacPrecErr= 2*np.finfo(float).eps
GaussFact = (nE*4*np.pi*w)/1.1

#Primitive Cell --------------------------------------------------------------------------------------------------------
b=nb*c #Size of lattice

# Potential
V=np.array([1])*(1+0*1j) #np.ones(nb)*(1+0*1j) #Potential vector (without defects)
# Masses
Mvec = np.ones(nb) # Mass matrix (without defects)
M = np.diag(Mvec, 0)

#Super Cell ------------------------------------------------------------------------------------------------------------
na=n*nb#Number of particles per cell
a= na*c #Size of lattice

#Potential
Vsc=np.tile(V,n)
#Masses
Mvecsc=np.tile(Mvec,n)

def invCumulFunc(graph,n,occpos,mu,var,x):
    print(occpos,mu,var,x)
    CumulFunc = np.zeros(n)
    DensProb = np.zeros(n)
    mu = np.append(np.append(mu - n,mu),mu+n) #periodicity considering effect of n+ is negligable
    if any(occpos == 0):
        CumulFunc[0] = 0
    else:
        CumulFunc[0] = np.sum(1 / np.pi * np.exp(-(mu) ** 2 / var))
    for i in range(1,n):
        if all(occpos != i):
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

    return pos

# Defects
if defects:
    availpos = np.arange(n)
    occpos = np.array([])

    for i in range(len(mval)):

        if dtype[i] == "ordered":
            pos = np.arange(i, n, int(1 / DefConc[i]))

            for j in range(nb):
                Mvecsc[nb*pos+j] = mval[i][j]
                Vsc[nb*pos+j] = kval[i][j]

        if dtype[i] == "random":
            ndefects = int(DefConc[i] * n)
            pos = rd.sample(list(availpos),ndefects)
            pos = np.array(pos)

            for p in pos:
                availpos = availpos[availpos!=p]

            for j in range(nb):
                Mvecsc[nb*pos+j] = mval[i][j]
                Vsc[nb*pos+j] = kval[i][j]

        if dtype[i] == "cluster":
            ndefects = int(DefConc[i] * n)
            mu = np.array([])
            graph = False
            for k in range(ndefects):

                if k==0:
                    pos = rd.sample(list(availpos),1)[0]
                else:
                    if k == ndefects-1:
                        graph = True
                    pos = invCumulFunc(graph,n,occpos,mu,clusterSize[i]**2,rd.random())
                mu = np.append(mu,pos)
                occpos = np.append(occpos,pos)
                for j in range(nb):
                    Mvecsc[nb*pos+j] = mval[i][j]
                    Vsc[nb*pos+j] = kval[i][j]

            for p in occpos:
                availpos = availpos[availpos != p]

Msc = np.diag(Mvecsc, 0)

#Namestamp

if defects:
    strdefects = "d%s"%('-'.join([str(i) for i in DefConc]))
    strdefects = strdefects + "_m%s"%('+'.join(['-'.join([str(j) for j in i]) for i in mval]))
    strdefects = strdefects + "_k%s"%('+'.join(['-'.join([str(j) for j in i]) for i in kval]))
else:
    strdefects = "d0"

if gaussian:
    strgauss = "g"
else:
    strgauss = "ng"

namestamp = "pm%s" %('-'.join([str(int(i)) for i in Mvec]))
namestamp = namestamp + "_k%s" %('-'.join([str(int(i)) for i in np.real(V)]))
namestamp = namestamp + strdefects + strgauss

print(Mvecsc)
print(Mvecsc[Mvecsc==2])
print(Mvecsc[Mvecsc==3])

# Solving for supercell at Q=0 -----------------------------------------------------------------------------------------

# K matrix
secdiag = Vsc[0:na - 1]
maindiag = np.hstack((-(Vsc[0] + Vsc[na - 1]), -(Vsc[1:na] + Vsc[0:na - 1])))
K = (np.diag(secdiag, -1) + np.diag(maindiag, 0) + np.diag(secdiag, 1))
K[na - 1, 0] = K[na - 1, 0] + Vsc[na - 1]
K[0, na - 1] = K[0, na - 1] + Vsc[na - 1]

#print(K)

# System M*w^2*x+Kx=0 => (M^-1*K)x = w^2x
sysmat = la.inv(Msc).dot(K)

print(np.diagonal(sysmat))

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
            eigenvecN[:, i] = eigenvecN[:, i] - eigenvecsc[:, i].dot(eigenvecN[:, j]) * eigenvecN[:, j]
    if ((eigenvecN[:, i] != eigenvecsc[:, i]).all):
        eigenvecN[:, i] = eigenvecN[:, i] / la.norm(eigenvecN[:, i])

eigenvecsc = np.array(eigenvecN)

qdisp=[]
omegadisp=[]
tol = omegasc.max()*1.1/(2*(nE-1)) #Tolerence for equality

# Solving for primitive cell at q = Q + G ------------------------------------------------------------------------------

#count=np.zeros((int(n/2)+1,na))

q=np.linspace(0,(2*np.pi/b)/n*np.ceil(n/2.0),np.ceil(n/2.0)+1)

#q=np.linspace(0,np.pi/b,int(n/2)+1)

dE = omegasc.max()*1.1/(nE-1)
w=w*omegasc.max()
if gaussian:
    Emin = -np.sqrt(2*w*dE*np.log(dE/(4*np.pi*w*CutOffErr**2)))
    nE=nE+int(np.ceil(-Emin/dE))
    Emin = np.ceil(Emin/dE)*dE
else:
    Emin = 0
E=np.linspace(Emin,omegasc.max()*1.1,nE)


Sf = np.zeros((nb,nE,int(n/2)+1))
Sftotal = np.zeros((nE,int(n/2)+1))

t_total_i = time.time()

for iq in range(int(n/2)+1): # Wave vector times lattice vector (1D) [-pi, pi]

    #timing
    t_q_i = time.time()

    #K matrix
    secdiag = V[0:nb-1]
    maindiag = np.hstack((-(V[0]+V[nb-1]),-(V[1:nb]+V[0:nb-1])))
    K = (np.diag(secdiag, -1) + np.diag(maindiag, 0) + np.diag(secdiag, 1))
    K[nb-1,0]=K[nb-1,0]+V[nb-1]*np.exp(-1j*q[iq]*b)
    K[0,nb-1]=K[0,nb-1]+V[nb-1]*np.exp(1j*q[iq]*b)

    #System M*w^2*x+Kx=0 => (M^-1*K)x = w^2x
    sysmat = la.inv(M).dot(K)

    #Solving system
    omegasq, eigenvec = la.eigh(-sysmat)

    omegasq[omegasq < 0] = 0
    omega=np.sqrt(omegasq)
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
    qdisp=np.hstack((qdisp,np.ones((1,nb))[0]*q[iq]))
    omegadisp=np.hstack((omegadisp,np.real(omega)))

    t_energy_loop = time.time()
    t_i_loop = np.zeros(nE)
    deltalist=np.zeros(len(E))
    for iE in range(nE):

        # timing
        t_i_i = time.time()
        t_l_loop = np.zeros(na)

        ############[ SUM ON ALL STATES ]############

        # sum i=1->na, outermost sum in definition of Sf
        for i in range(na):

            ############[ DIRAC DELTA FUNCTION ]############

            # delta (E - epsi)
            # print(omegasc[i])
            # print(dE)
            # print(delta)
            if gaussian == True:
                delta = np.sqrt(dE / (4 * np.pi * w)) * np.exp(-(omegasc[i] - E[iE]) ** 2 / (4 * w * dE))
                condition = (delta > CutOffErr)
            else:
                condition = (abs(omegasc[i] - E[iE]) < tol)
                delta = 1
            if condition:

                deltalist[iE] = deltalist[iE] + delta

                ScalarProd = np.zeros(nb)*(1+0*1j)
#                count[iq,i] = count[iq,i] + 1
                #timing
                t_l_i = time.time()

                ############[ SCALAR PRODUCT ]############

                # sum l=1->Nt on all space for the scalar product (in this case Nc = 1 => Nt = na)
                for l in range(Nc*na):
                    # sum s=1->nb on all solutions of the primittive cell
                    # TODO: Change np.floor(l/b) to (l/nb)
                    for s in range(nb):
                        ScalarProd[s] = ScalarProd[s] + 1/Nc*np.conj(eigenvecsc[l%na,i])*1/np.sqrt(n)*eigenvec[l%nb,s]*np.exp(-1j*q[iq]*np.floor(l/nb)*b)

                # ////////////[ SCALAR PRODUCT ]////////////
                for s in range(nb):
                    Sf[s, iE, iq] = Sf[s, iE, iq] + delta * abs(ScalarProd[s]) ** 2
                Sftotal[iE, iq] = Sftotal[iE, iq] + delta * abs(np.sum(ScalarProd)) ** 2

                #timing
                t_l_loop[i] = t_l_loop[i] + time.time()-t_l_i

            #////////////[ DIRAC DELTA FUNCTION ]////////////

        # ////////////[ SUM ON ALL STATES ]////////////

        #timing
        t_i_loop[iE] = t_i_loop[iE]+time.time()-t_i_i
#        print('Total times for Loop in l=', t_l_loop)

    #timing
    print('Total times for Loop in i=', t_i_loop)
    print(iq)
    print('(',iq/(n/2.0)*100.0,'%)')
    t_q_f = time.time()
    print('Total iq=',t_q_f - t_q_i)
    print('Energy loop time=',t_q_f-t_energy_loop)
    print('System solve + GS proces=', -t_q_i + t_energy_loop)

    print('-------------------------------------------------')

t_total_f = time.time()
print(t_total_f - t_total_i)

plt.figure()
plt.plot(qdisp,omegadisp,'.')
plt.savefig(folder+"primitive_band_"+namestamp+".png")

Sf[Sf<MacPrecErr]=0

plt.figure()
plt.imshow(Sftotal, interpolation='None', origin='lower',
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
print('Gaussian Factor')
print(GaussFact)

#Display
plt.figure()
plt.plot(E, Sf[:,:, 0].T, '.-')
ax = plt.gca()
ymin, ymax = ax.get_ylim()
deltalist = deltalist / max(deltalist)
for npE in range(len(E)):
    ax.vlines(x=E[npE], ymin=ymin, ymax=ymax, color='r', alpha=deltalist[npE])
plt.ylim([ymin, ymax])
plt.plot(E,deltalist*ymax, color='r')

plt.savefig(folder+"slice_0_"+namestamp+".png")

#lifetime calculations -------------------------------------------------------------------------------------------------

EAvg = np.zeros((nb, int(n / 2) + 1))
EsqAvg = np.zeros((nb, int(n / 2) + 1))
Var = np.zeros((nb, int(n / 2) + 1))


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
plt.ylabel(r'Angular Frequency($\omega$)')
plt.xlabel('Wave vector (q)')
plt.title("Average frequency and standard deviation")
#plt.savefig(folder+"spectral_map_"+namestamp+".png")

plt.figure()
plt.plot(q,Tau.T)
plt.ylabel(r'Lifetime($\tau$)')
plt.xlabel('Wave vector (q)')
plt.title("Lifetime")
#plt.savefig(folder+"spectral_map_"+namestamp+".png")

# Lattice Thermal Conductivity -----------------------------------------------------------------------------------------

dq = 2*np.pi/a
bar = hbar_kb*EAvg/T
C=np.ones((nb,int(n / 2) + 1))
C[bar>=MacPrecErr] = bar[bar>=MacPrecErr]**2*np.exp(bar[bar>=MacPrecErr])/(np.exp(bar[bar>=MacPrecErr])-1)**2
v = np.zeros((nb,int(n / 2) + 1))
v[:,1:-1] = 1/dq*1/2*(EAvg[:,2:]-EAvg[:,:-2])
v[:,0] = 1/dq*(2*EAvg[:,1]-3/2*EAvg[:,0]-1/2*EAvg[:,2])
v[:,-1] = 1/dq*(-2*EAvg[:,-2]+3/2*EAvg[:,-1]+1/2*EAvg[:,-3])

k = dq*np.sum(C*v**2*Tau,1)

print("##################################################")
print("Total Lattice Thermal Conductivity:", np.sum(k))
for s in range(nb):
    print("Band %d contribution:"%s, k[s])
print("--------------------------------------------------")
print("Prameters:")
print("Specific Heat (C):", C)
print("Group velocity (v):", v)
print("Relaxation Time (Tau):", Tau)
print("##################################################")

plt.show()
print('done')
