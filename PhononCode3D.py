import numpy as np
import matplotlib
import numpy.linalg as la
import pickle
import prepare
from mpi4py import MPI
import time
from pylada.crystal import supercell

# Display
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

# try:
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
# except:
#     pass

# Initial MPI calls
comm = MPI.COMM_WORLD
master = 0
n_proc = comm.Get_size()
rank = comm.Get_rank()

# Constant                                                                              
hbar_kb = 7.63824*10**(-12) #second kelvins

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

def resetColors(plt):
    try:
        plt.gca().set_prop_cycle(None)
    except:
        plt.gca().set_color_cycle(None)

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
tol_q = 1e-3
T = 300

possc = np.floor(np.array([a.pos.tolist() for a in Asc]).dot(icell)+tol).dot(A.cell) # Position in supercell

Q = np.array(qsc)
q = [np.array(v)*2*np.pi for v in q]
# q[:8] = np.array(prepare.on_path(path, rsc, irsc))[:,:3]*np.pi*2 #TMP

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
    ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), A.cell[:,0], A.cell[:,1], A.cell[:,2], color = 'red')
    ax.view_init(0,90)
    fig.savefig("integral.png")

    q_path, pos_on_path, syms = prepare.on_path_plot(path, rsc, irsc)
    q_path = np.array(q_path)[:,:3]*np.pi*2
    
    q_index = []
    for j in range(np.shape(q_path)[0]):
        for i in range(nq):
            if la.norm(q_path[j,:] - q[i]) <= tol_q:
                q_index.append(i)

    q_length = np.array(pos_on_path)
    omegadisp = np.zeros((len(q_index), npc))
    plot_n = 100
    Sf_plot = np.zeros((npc, nE, plot_n))
    linq = np.linspace(0,syms[-1],plot_n)
    for i,ind in enumerate(q_index):
        omegadisp[i,:] = omegapc[ind]
        Sf_plot[:,:,np.where(linq <= q_length[i])[0][-1]] = Sf[:,:,ind]    

    plt.figure()
    plt.plot(q_length,omegadisp,'.')
    plt.plot(q_length,omegadisp,'k', alpha = 0.5)
    plt.imshow(np.sum(Sf_plot,0), interpolation='None', origin='lower',
               cmap=plt.cm.nipy_spectral_r,aspect='auto',extent=[0, syms[-1], E.min(), E.max()],vmax=1, vmin=0)
    plt.xlim((0,syms[-1]))

    # Location of highsymmetry points along the path
    ax = plt.gca()
    ax.set_xticks(syms)
    ax.set_xticklabels([r'\Gamma','K\|U','X',r'\Gamma','L','K\|U','W','X','W','L'])
    
    plt.title('Phonon dispersion relation for pure Ge')
    plt.xlabel(r'High symmetry q-points')
    plt.ylabel(r'Frequency (THz)')
    plt.margins(0,0)

    # Add dashed lines
    for sym in syms[1:]:
        plt.plot([sym,sym],ax.get_ylim(),'--k',alpha=0.5)

    plt.savefig(folder+"spectral_vs_primittive.png")

    plt.figure()
    plt.imshow(np.sum(Sf_plot,0), interpolation='None', origin='lower',
               cmap=plt.cm.nipy_spectral_r,aspect='auto',extent=[0, syms[-1], E.min(), E.max()],vmax=1, vmin=0)
    plt.xlabel(r'High symmetry q-points')
    plt.ylabel(r'Frequency (THz)')
    plt.title("Total band")
    plt.savefig(folder+"spectral_map.png")


    for s in range(npc):
        plt.figure()
        plt.imshow(Sf_plot[s,:,:], interpolation='None', origin='lower',
                   cmap=plt.cm.nipy_spectral_r,aspect='auto',extent=[0, syms[-1], E.min(), E.max()],vmax=1, vmin=0)
        plt.ylabel(r'Angular Frequency($\omega$)')
        plt.xlabel('Wave vector(q)')
        plt.title("Band %d"%s)
        plt.savefig(folder+"band_%d_spectral_map.png"%s)

if rank == master:
    
    ############[ Life time and Averages ]############
    EAvg = np.zeros((npc, len(q)))
    EsqAvg = np.zeros((npc, len(q)))
    Var = np.zeros((npc, len(q)))

    for s in range(npc):
        for iE in range(nE):
            EAvg[s,:] = EAvg[s,:] + E[iE]*Sf[s,iE,:]
            EsqAvg[s,:] = EsqAvg[s,:] + (E[iE]**2)*Sf[s,iE,:]
        EAvg[s,:] = EAvg[s,:]/np.sum(Sf[s, :, :], 0)
        EsqAvg[s,:] = EsqAvg[s,:]/np.sum(Sf[s, :, :], 0)

    Var = EsqAvg-EAvg**2
    DeltaE = np.sqrt(Var)
    Tau = 1/DeltaE

     # ////////////[ Life time and Averages ]////////////

    if output:

        # Average and standard deviation plot ------------
        plt.figure()
        plt.plot(q_length,EAvg[:,q_index].T)
        resetColors(plt)
        for s in range(npc):
            plt.fill_between(q_length, EAvg[s, q_index] - DeltaE[s, q_index], EAvg[s, q_index] + DeltaE[s, q_index], alpha=0.2)
        resetColors(plt)
        plt.plot(q_length,omegadisp,'--')
        plt.xlim((0,syms[-1]))

        # Location of highsymmetry points along the path
        ax = plt.gca()
        ax.set_xticks(syms)
        ax.set_xticklabels([r'\Gamma','K\|U','X',r'\Gamma','L','K\|U','W','X','W','L'])
    
        plt.title('Average bands and standard deviation along high symmetry path')
        plt.xlabel(r'High symmetry q-points')
        plt.ylabel(r'Frequency (THz)')
        plt.margins(0,0)

        # Add dashed lines
        for sym in syms[1:]:
            plt.plot([sym,sym],ax.get_ylim(),'--k',alpha=0.5)

        plt.savefig(folder+"avg_std.png")

        # Lifetime plot (vs q)------------

        plt.figure()
        plt.plot(q_length,Tau[:,q_index].T)
        plt.xlim((0,syms[-1]))

        # Location of highsymmetry points along the path
        ax = plt.gca()
        ax.set_xticks(syms)
        ax.set_xticklabels([r'\Gamma','K\|U','X',r'\Gamma','L','K\|U','W','X','W','L'])
        plt.margins(0,0)

        # Add dashed lines
        for sym in syms[1:]:
            plt.plot([sym,sym],ax.get_ylim(),'--k',alpha=0.5)

        plt.ylabel(r'Lifetime(ps)')
        plt.xlabel(r'High symmetry q-points')
        plt.title("Lifetime along high symmetry path")
        plt.savefig(folder+"tau_q.png")

        plt.figure()
        plt.plot(EAvg.T, Tau.T, '.')
        plt.ylabel(r'Lifetime(ps)')
        plt.xlabel(r'Frenquency (THz)')
        plt.title("Lifetime as a function of frequency (on all q-points)")
        plt.savefig(folder+"tau_w.png")

    ############[ Lattice Thermal Conductivity ]############

    rpc = prepare.reciprocal(A.cell) #reciprocal lattice
    irpc = np.linalg.inv(rpc) #inverse of reciprocal lattice

    q_all = prepare.all_points(rpc, irpc, rsc, irsc)
    q_all = np.array(q_all)[:,:3]*np.pi*2

    q_integral = []
    for j in range(np.shape(q_all)[0]):
        for i in range(nq):
            if la.norm(q_all[j,:] - q[i]) <= tol_q:
                q_integral.append(i)

    Vol = la.det(A.cell)
    dq = la.norm(rsc,axis=1)
    bar = 2 * np.pi * hbar_kb * EAvg[:,q_index+q_integral]/T
    C = np.ones((npc,len(q_index+q_integral)))
    C[bar>=MacPrecErr] = bar[bar>=MacPrecErr]**2 \
    * np.exp( bar[bar>=MacPrecErr] ) / ( np.exp( bar[bar>=MacPrecErr] ) - 1 )**2
    
    # Velocities
    v = np.zeros((npc,len(q_index + q_integral),3))
    for iq, q_int in enumerate(q_index + q_integral):
        qp = q[q_int]
        # for each dimention...
        for i in range(3):
            q_plus = qp + rsc[i,:]
            q_minus = qp - rsc[i,:]
            q_p_ind = None
            q_m_ind = None
            # ... checks if the neirest neighbors are in the list
            for j in range(nq):
                if la.norm(q_plus - q[j]) <= tol_q:
                    q_p_ind = j
                if la.norm(q_minus - q[j]) <= tol_q:
                    q_m_ind = j
            # If both neigbors are present calculates centered finite diff
            if q_p_ind != None and q_m_ind !=None:
                v[:,iq,:] += 1/dq[i]*1/2*(EAvg[:,q_p_ind:q_p_ind+1]-EAvg[:,q_m_ind:q_m_ind+1]).dot(rsc[i:i+1,:])
            
            # If right is present...
            elif q_p_ind != None and q_m_ind ==None:
                q_plus2 = qp + 2*rsc[i,:]
                q_p2_ind = None
                # check if the second nearest is present
                for j in range(nq):
                    if la.norm(q_plus2 - q[j]) <= tol_q:
                        q_p2_ind = j
                if q_p2_ind != None:
                    v[:,iq,:] += 1/dq[i]*(2*EAvg[:,q_p_ind:q_p_ind+1]-3/2*EAvg[:,q_int:q_int+1]-1/2*EAvg[:,q_p2_ind:q_p2_ind+1])
                else:
                    raise(RuntimeError("The q-grid is incomplete!"))
            elif q_p_ind == None and q_m_ind != None:
                q_minus2 = qp - 2*rsc[i,:]
                q_m2_ind = None
                for j in range(nq):
                    if la.norm(q_minus2 - q[j]) <= tol_q:
                        q_m2_ind = j
                if q_m2_ind != None:
                    v[:,iq,:] += 1/dq[i]*(-2*EAvg[:,q_m_ind:q_m_ind+1]+3/2*EAvg[:,q_int:q_int+1]+1/2*EAvg[:,q_m2_ind:q_m2_ind+1])
                else:
                    raise(RuntimeError("The q-grid is incomplete!"))
            else:
                print iq, q_int, q[q_int]
                raise(RuntimeError("The q-grid is incomplete!"))
    
    k = np.zeros((npc,len(q_index + q_integral),3,3))
    for i_npc, v_npc in enumerate(v):
        for i_q, v_q in enumerate(v_npc):
            print np.reshape(v_q,(3,1)).dot(np.reshape(v_q,(1,3)))
            print "---"
            k[i_npc,i_q,:,:] = (2*np.pi)**3/Vol * np.sum(C[i_npc,i_q]*np.reshape(v_q,(3,1)).dot(np.reshape(v_q,(1,3)))*Tau[i_npc,i_q],1)

    LTC = np.sum(np.sum(k[:,q_integral,:,:], axis = 0),axis = 0)

    k_plot = np.trace(k, axis1=2, axis2=3)

    if output:
        # Lifetime plot (vs q)------------

        plt.figure()
        plt.plot(q_length,k_plot[:,q_index].T)
        plt.xlim((0,syms[-1]))
        
        # Location of highsymmetry points along the path
        ax = plt.gca()
        ax.set_xticks(syms)
        ax.set_xticklabels([r'\Gamma','K\|U','X',r'\Gamma','L','K\|U','W','X','W','L'])
        plt.margins(0,0)
        
        # Add dashed lines
        for sym in syms[1:]:
            plt.plot([sym,sym],ax.get_ylim(),'--k',alpha=0.5)
            
        plt.ylabel(r'LTC (a.u.)')
        plt.xlabel(r'High symmetry q-points')
        plt.title("Lifetime along high symmetry path")
        plt.savefig(folder+"k_q.png")

        print "##################################################"
        print "Total Lattice Thermal Conductivity:", LTC 
        for s in range(npc):
            print "Band contribution:", np.sum(k[:,q_integral,:,:], axis = 1)[s]
        print "##################################################"
            
    # ////////////[ Lattice Thermal Conductivity ]////////////

exec(open("test_ortho.py").read())
