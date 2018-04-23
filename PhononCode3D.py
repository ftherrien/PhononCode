import numpy as np
import matplotlib
import numpy.linalg as la
import pickle
import prepare
from mpi4py import MPI
import time

# Display
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

# Initial MPI calls
comm = MPI.COMM_WORLD
master = 0
n_proc = comm.Get_size()
rank = comm.Get_rank()

# Errors
MacPrecErr= 2*np.finfo(float).eps

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

# Begin program ===============================================================================

# PARAMETERS \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/

nE = 100
w = 0.01
CutOffErr=10**-4
output = True
folder = "outdir"
gaussian = False

# \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\//\/\/\/\/\/\

# Importing data from prepare.py
qsc, omegasc, eigenvecsc = pickle.load(open("eigsc.dat"))
q, omega, eigenvec = pickle.load(open("eigpc.dat"))
A, Asc = pickle.load(open("structures.dat"))
path = pickle.load(open("path.dat"))

# q = q[0:8]
# omega = omega[0:8]
# eigenvec = eigenvec[0:8]

rsc = prepare.reciprocal(Asc.cell) #reciprocal lattice
irsc = np.linalg.inv(rsc) #inverse of reciprocal lattice
icell = np.linalg.inv(A.cell) #Inverse of primitive lattice

tol = 1e-12

possc = np.floor(np.array([a.pos.tolist() for a in Asc]).dot(icell)+tol).dot(A.cell) # Position in supercell

Q = np.array(qsc)
q = [np.array(v)*2*np.pi for v in q]
q[:8] = np.array(prepare.on_path(path, rsc, irsc))[:,:3]*np.pi*2 #TMP

nq = len(q)
if rank == master and output:
    print "Started calculation with %d q points on %d cores"%(nq, n_proc)

nsc = len(omegasc[0])

# Gram-Schmidt Process (modified considering most vectors are orthonormal)     
# Super Cell
eigenvecN = np.array(eigenvecsc[0]).T
eigenvecsc = np.array(eigenvecsc[0]).T
omegasc = np.array(omegasc[0])
for i in range(nsc):
    for j in np.arange(i+1,nsc):
        if (abs(omegasc[i] - omegasc[j]) < MacPrecErr):
            eigenvecN[:, i] = eigenvecN[:, i] - eigenvecsc[:, i].dot(np.conj(eigenvecN[:, j])) * eigenvecN[:, j]
    if ((eigenvecN[:, i] != eigenvecsc[:, i]).all):
        eigenvecN[:, i] = eigenvecN[:, i] / la.norm(eigenvecN[:, i])
                
eigenvecsc = np.array(eigenvecN)


# Primitive Cell
omegapc = []
eigenvecpc = []
for k in range(nq):
    eigenvecN = np.array(eigenvec[k]).T
    eigenvecpc.append(np.array(eigenvec[k]).T)
    omegapc.append(np.array(omega[k]))
    npc = len(omega[-1])
    for i in range(npc):
        for j in np.arange(i+1,npc):
            if (abs(omegapc[-1][i] - omegapc[-1][j]) < MacPrecErr):
                eigenvecN[:, i] = eigenvecN[:, i] - eigenvecpc[-1][:, i].dot(np.conj(eigenvecN[:, j])) * eigenvecN[:, j]
        if ((eigenvecN[:, i] != eigenvecpc[-1][:, i]).all):
            eigenvecN[:, i] = eigenvecN[:, i] / la.norm(eigenvecN[:, i])
            
    eigenvecpc[-1] = np.array(eigenvecN)    

# Defining the energy range 
MaxOmegasc = omegasc.max()*1.1
                             
dE =  MaxOmegasc/(nE-1)
sig = w*MaxOmegasc

if gaussian:
    Emin = - sig * np.sqrt( np.log( dE / ( CutOffErr * sig * np.sqrt( np.pi ) ) ) )
else:
    Emin = 0
nE = nE+int(np.ceil(-Emin/dE))
Emin = np.floor(Emin/dE)*dE

E=np.linspace(Emin,MaxOmegasc,nE)
normalize = 1/np.sqrt(nsc/npc)

# Load balancing the jobs
Local_range = load_balance(nq)

Local_Sf = np.zeros((npc,nE,len(Local_range)))
Local_Sftotal = np.zeros((nE,len(Local_range)))

############[ SPECTRAL FUNCTION ]############
for jq,iq in enumerate(Local_range):
    t_q_i = time.time()
    deltalist=np.zeros(nE)
    for iE in range(nE):
         snorm = np.zeros(3)*(1+0*1j)
         for i in range(nsc):
             if gaussian:
                 delta = dE / (sig*np.sqrt(np.pi)) * np.exp(-(omegasc[i] - E[iE]) ** 2 / sig**2)
                 condition = delta > CutOffErr
             else:
                 delta = 1
                 condition = abs(omegasc[i] - E[iE]) < dE/2
             if condition:
                 deltalist[iE] = deltalist[iE] + delta
                 ScalarProd = np.zeros(npc)*(1+0*1j)
                 for l in range(nsc):
                     for s in range(npc):
                         ScalarProd[s] = ScalarProd[s] + normalize*np.conj(eigenvecsc[l,i])*eigenvecpc[iq][l%npc,s]*np.exp(1j*q[iq].dot(possc[l//3]))
                     
                 for s in range(npc):
                     Local_Sf[s, iE, jq] = Local_Sf[s, iE, jq] + delta * abs(ScalarProd[s]) ** 2
    t_q_f = time.time()
    if output:
        print "Finished %d in core %d in %f seconds"%(iq,rank,t_q_f - t_q_i)
# ////////////[ SPECTRAL FUNCTION ]////////////

Sf = np.concatenate(comm.allgather(Local_Sf),2)

# Sf = pickle.load(open(folder+"Sf.dat")) # TMP

Sf[Sf<MacPrecErr]=0

# Save the spectral function
pickle.dump( Sf, open(folder+"Sf.dat", "wb" ) )

if output and rank == master:

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    integral = np.array(possc)
    ax.scatter(integral[:,0], integral[:,1], integral[:,2])
    ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), Asc.cell[:,0], Asc.cell[:,1], Asc.cell[:,2])
    ax.view_init(0,0)
    fig.savefig("integral.png")

    q_path, pos_on_path, length_of_path = prepare.on_path_non_unique(path, rsc, irsc)
    q_path = np.array(q_path)[:,:3]*np.pi*2

    # TODO: Adjust length on path to be the right length on the complete path
    
    q_index = []
    for j in range(np.shape(q_path)[0]):
        for i in range(nq):
            if la.norm(q_path[j,:] - q[i]) <= 1e-3:
                q_index.append(i)

    q_length = np.array(pos_on_path)
    omegadisp = np.zeros((len(q_index), npc))
    Sf_plot = np.zeros((npc,nE,len(q_index)))
    for i,ind in enumerate(q_index):
        omegadisp[i,:] = omegapc[ind]
        Sf_plot[:,:,i] = Sf[:,:,ind]

    plt.figure()
    plt.plot(q_length,omegadisp,'.-')
    plt.ylabel(r'Angular Frequency($\omega$)')
    plt.xlabel('Wave vector(q)')
    plt.title("Primitive cell band structure")
    plt.savefig(folder+"primitive_band.png")

    plt.figure()
    plt.imshow(np.sum(Sf_plot,0), interpolation='None', origin='lower',
               cmap=plt.cm.nipy_spectral_r,aspect='auto',extent=[q_length.min(), q_length.max(), E.min(), E.max()],vmax=1, vmin=0)
    plt.ylabel(r'Angular Frequency($\omega$)')
    plt.xlabel('Wave vector(q)')
    plt.title("Total band")
    plt.savefig(folder+"spectral_map.png")


    for s in range(npc):
        plt.figure()
        plt.imshow(Sf_plot[s,:,:], interpolation='None', origin='lower',
                   cmap=plt.cm.nipy_spectral_r,aspect='auto',extent=[q_length.min(), q_length.max(), E.min(), E.max()],vmax=1, vmin=0)
        plt.ylabel(r'Angular Frequency($\omega$)')
        plt.xlabel('Wave vector (q)')
        plt.title("Band %d"%s)
        plt.savefig(folder+"band_%d_spectral_map.png"%s)

    
