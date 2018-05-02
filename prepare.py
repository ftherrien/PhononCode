from pythonQE import pwcalc, phcalc, q2rcalc, matcalc, dyncalc, submit_jobs
from copy import deepcopy
import os
from pylada.crystal import supercell, Structure
import pylada.periodic_table as pt
import pickle
import numpy as np

tol = 1e-10

def reciprocal(cell):
    rec = deepcopy(cell)
    V = np.linalg.det(cell)
    for i in range(3):
        rec[i,:] = 2*np.pi/V*np.cross(cell[(i+1)%3,:], cell[(i+2)%3,:])
    return rec

def explicit_path(path):
    epath = []
    for i,line in enumerate(path[:-1]):
        for j in range(len(line))[:-1]:
            if j == 0:
                linpath = np.reshape(np.linspace(line[j],path[i+1][j],int(line[-1])+1), (1,int(line[-1])+1))
            else:
                linpath = np.concatenate((linpath, np.reshape(np.linspace(line[j],path[i+1][j],int(line[-1])+1), (1,int(line[-1])+1))), axis=0)
        linpath = np.concatenate((linpath, np.ones((1,int(line[-1])+1))),axis=0)
        if i == 0:
            for j in range(np.shape(linpath)[1]):
                epath.append(list(linpath[:,j]))
        else:
            for j in range(np.shape(linpath)[1]-1):
                epath.append(list(linpath[:,j+1]))
    return epath

def unique(closest):
    unique = []
    for line in closest:
        there = False
        for check in unique:
            if all(check == line):
                there = True
                break
        if not there:
            unique.append(line)
    return np.array(unique)

def approx(i, vec):
    if i:
        return np.ceil(vec-tol)
    else:
        return np.floor(vec+tol)

def closest_box(path, rsc, irsc):
    epath = np.array(explicit_path(path))*np.pi*2 # explicit_path
    epathrc = epath[:,0:3].dot(irsc) # explicit_reciprocal_path

    closest = [] 
    for i in range(2):
        for j in range(2):
            for k in range(2):
                closest.append(np.concatenate((approx(i,epathrc[:,0:1]), approx(j,epathrc[:,1:2]), approx(k,epathrc[:,2:3])), axis=1))

    newpath = unique(np.concatenate(closest, axis=0)).dot(rsc)/ (np.pi*2)

    return np.concatenate((newpath, np.ones((np.shape(newpath)[0],1))), axis=1).tolist() #Adding ones for QE

def on_path(path, rsc, irsc):
    epath = np.array(explicit_path(path))*np.pi*2
    epathrc = epath[:,0:3].dot(irsc)
    
    npts = np.array(path)[:,3]
    path = np.array(path)[:,:3]

    closest_points = unique(np.round(epathrc)).dot(rsc) / (2*np.pi)
    onpath = []
    for i,v in enumerate(path[:-1,:]):
        if npts[i] > 1:
            for p in closest_points:
                isonpath = True
                t = []
                for j in range(3):
                    if abs((path[i+1][j]-v[j])) >= tol: # make sure there is no division by zero
                        t.append((p[j]-v[j])/(path[i+1][j]-v[j]))
                    else:
                        if abs((p[j]-v[j])) >= tol: # if so, check if nominator is near 0
                            isonpath = False
            
                # Makes sure the multiplier t is the same for each component
                if len(t) == 2:
                    if abs(t[1] - t[0]) >= tol:
                        isonpath = False
                elif len(t) == 3:
                    for j in range(3):
                        if abs(t[j]-t[(j+1)%3]) >= tol:
                            isonpath = False
            
                # Includes the last point on the path
                if i == np.shape(path)[0] - 2:
                    mul = 1
                else:
                    mul = -1
            
                # Makes sure it is between the 2 points
                if any(np.array(t) > 1 + mul*tol) or any(np.array(t) < 0 - tol):
                    isonpath = False
            
                # Adds the points that passed all the tests on the path
                if isonpath:
                    onpath.append(list(p))

    newpath = unique(np.array(onpath)) 
         
    return np.concatenate((newpath, np.ones((np.shape(newpath)[0],1))), axis=1).tolist() #Adding ones for QE
         

def on_path_plot(path, rsc, irsc):
    epath = np.array(explicit_path(path))*np.pi*2
    epathrc = epath[:,0:3].dot(irsc)
    
    npts = np.array(path)[:,3]
    path = np.array(path)[:,:3]
    
    closest_points = unique(np.round(epathrc)).dot(rsc) / (2*np.pi)
    onpath = []
    dist_on_path = 0
    syms = [0]
    pos_on_path = []
    for i,v in enumerate(path[:-1,:]):
        if npts[i] > 1:
            for p in closest_points:
               isonpath = True
               t = []
               for j in range(3):
                   if abs((path[i+1][j]-v[j])) >= tol: # make sure there is no division by zero
                       t.append((p[j]-v[j])/(path[i+1][j]-v[j]))
                   else:
                       if abs((p[j]-v[j])) >= tol: # if so, check if nominator is near 0
                           isonpath = False
            
               # Makes sure the multiplier t is the same for each component
               if len(t) == 2:
                   if abs(t[1] - t[0]) >= tol:
                       isonpath = False
               elif len(t) == 3:
                   for j in range(3):
                       if abs(t[j]-t[(j+1)%3]) >= tol:
                           isonpath = False
            
               # Includes the last point on the path
               if i == np.shape(path)[0] - 2:
                   mul = 1
               else:
                   mul = -1
            
               # Makes sure it is between the 2 points
               if any(np.array(t) > 1 + mul*tol) or any(np.array(t) < 0 - tol):
                   isonpath = False
            
               # Adds the points that passed all the tests on the path
               if isonpath:
                   onpath.append(list(p))
                   pos_on_path.append(dist_on_path + np.linalg.norm(p-v))

            dist_on_path += np.linalg.norm(path[i+1]-v)
            syms.append(dist_on_path)
                 
    newpath = np.array(onpath) 
         
    return (np.concatenate((newpath, np.ones((np.shape(newpath)[0],1))), axis=1).tolist(),
            pos_on_path, syms) #Adding ones for QE

def all_points(rpc, irpc, rsc, irsc):
    # big square
    # Finds the furthest corner
    corner = np.array([[0,0,0]])
    for i in range(2):
        for j in range(2):
            for k in range (2):
                corner = np.reshape(np.max(np.concatenate([abs(np.array([[i,j,k]]).dot(rpc).dot(irsc)), corner], axis=0), axis=0),(1,3))
    
    corner = abs(np.ceil(corner)).astype(int)
    
    list_in = []
    for i in range(-corner[0,0], corner[0,0]):
        for j in range(-corner[0,1], corner[0,1]):
            for k in range (-corner[0,2], corner[0,2]):
                p = np.array([i,j,k]).dot(rsc)
                if (p.dot(irpc) >= -0.5 - tol).all() and (p.dot(irpc) < 0.5 - tol).all():
                    list_in.append(p.tolist())
    
    list_in = np.array(list_in) / (2*np.pi)

    return np.concatenate((list_in, np.ones((np.shape(list_in)[0],1))), axis=1).tolist() #Adding ones for QE

def derivative_points(path, rsc):
    q = np.array(explicit_path(path))[:,:3] * 2 * np.pi
    
    coef = np.array([-2, -1, 1, 2])
    
    for iq, qp in enumerate(q):
        # For each dimention... 
        for i in range(3):
            q_disp = np.zeros((4,3))
            q_bool = [False]*4
            for k, mul in enumerate(coef):
                # ...finds the 2 nearest neighbors in each direction
                q_disp[k,:] = qp + mul*rsc[i,:]
    
            # Check if the neighbors are already in the list
            for j in range(len(q)):
                for k, mul in enumerate(coef):
                    if np.linalg.norm(q_disp[k,:] - q[j,:]) <= tol:
                        q_bool[k] = True
    
            # If none of the nieghbors is present...
            if not any(q_bool):
                # ..adds the symmetric ones
                q = np.concatenate((q,q_disp[1:3,:]), axis = 0)

            # If one or more neighbor is present but not next to eachother
            elif not any([a and b for a, b in zip(q_bool[:3], q_bool[1:])]):
                # finds the most centered neighbor
                q_ind = np.argmin(abs(coef[q_bool]))
                q_ind = np.where(q_bool)[0][q_ind]
                # Adds the symmetric neighbor in priority
                if q_ind < 2:
                    q = np.concatenate((q,[q_disp[q_ind+1,:]]), axis = 0)
                else:
                    q = np.concatenate((q,[q_disp[q_ind-1,:]]), axis = 0)
    
    return np.concatenate((q/(2*np.pi), np.ones((np.shape(q)[0],1))), axis=1).tolist()

def to_relaxed_coord(path, irpc_perfect, rpc):
    weights = np.array(path)[:,3:4]
    path = np.array(path)[:,:3].dot(irpc_perfect).dot(rpc)
    return np.concatenate((path, weights), axis=1).tolist()


# Begin program ======================================================================
if __name__ == "__main__":
    nproc = 64
    
    # Primittive structure
    perfectStruc = Structure([[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5]])
    perfectStruc.add_atom(0,0,0,'Si')
    perfectStruc.add_atom(0.25,0.25,0.25,'Si')
    
    # Building the (perfect) supercell
    perfectStrucsc = supercell(perfectStruc,[[3,0,0],[0,3,0],[0,0,3]]);
        
    # Primitive Cell Calculations ########################################################
    
    # Relaxation
    pwrelax = pwcalc()
    pwrelax.name = "pc"
    pwrelax.calc_type = "vc-relax"
    pwrelax.restart_mode = "from_scratch"
    pwrelax.pseudo_dir = os.path.expanduser("~/scratch/pseudo_pz-bhs/")
    pwrelax.celldm = 10.7
    pwrelax.ecutwfc = 45.0
    pwrelax.ecutrho = 400.0
    pwrelax.nbnd = len(perfectStruc)*4
    pwrelax.occupations = "fixed"
    pwrelax.masses = {'Si':pt.Si.atomic_weight}
    pwrelax.from_pylada(perfectStruc)
    pwrelax.kpoints = [8,8,8]
    
    # submit_jobs(pwrelax, np = nproc)
    ene = pwrelax.read_energies()
    while (abs(ene[-1] - ene[-2]) > 1e-8):
        pwrelax.atomic_pos = pwrelax.read_atomic_pos()
        pwrelax.cell = pwrelax.read_cell()
        submit_jobs(pwrelax, np = nproc)
        ene = pwrelax.read_energies()
    
    # Self consistant run
    pwscf = deepcopy(pwrelax)
    pwscf.calc_type = 'scf'
    pwscf.atomic_pos = pwrelax.read_atomic_pos()
    pwscf.cell = pwrelax.read_cell()
    
    # Phonons
    ph = phcalc()
    
    ph.name = pwscf.name
    ph.masses = pwscf.masses
    ph.qpoints = [6,6,6]
    
    # Inverse Fourier transform
    q2r = q2rcalc()
    
    q2r.name = pwscf.name
    
    # Fourier transform
    matdyn = matcalc()
    
    matdyn.name = pwscf.name
    matdyn.masses = pwscf.masses
    
    # Setting cells and inverses
    Struc = pwscf.to_pylada()

    ippc = np.linalg.inv(perfectStruc.cell)
    Strucsc = perfectStrucsc.cell.dot(ippc).dot(Struc.cell)

    Strucsc = supercell(Struc, Strucsc)

    pickle.dump((Struc, Strucsc), open("structures.dat","wb"))

    rsc = reciprocal(Strucsc.cell) #reciprocal lattice
    irsc = np.linalg.inv(rsc) #inverse of reciprocal lattice

    rpc = reciprocal(Struc.cell) #reciprocal lattice
    irpc = np.linalg.inv(rpc) #inverse of reciprocal lattice

    rpc_prefect = reciprocal(perfectStruc.cell)
    irpc_perfect = np.linalg.inv(rpc_prefect)
    
    # q-path for primittive cell
    path = [
        [0.0000000,   0.0000000,   0.0000000, 10],
        [0.7500000,   0.7500000,   0.0000000, 1 ],
        [0.2500000,   1.0000000,   0.2500000, 10],
        [0.0000000,   1.0000000,   0.0000000, 10],
        [0.0000000,   0.0000000,   0.0000000, 10],
        [0.5000000,   0.5000000,   0.5000000, 10],
        [0.7500000,   0.7500000,   0.0000000, 1 ],
        [0.2500000,   1.0000000,   0.2500000, 10],
        [0.5000000,   1.0000000,   0.0000000, 10],
        [0.0000000,   1.0000000,   0.0000000, 10],
        [0.5000000,   1.0000000,   0.0000000, 10],
        [0.5000000,   0.5000000,   0.5000000, 1 ]]

    path = to_relaxed_coord(path, irpc_perfect, rpc)

    epath = np.array(explicit_path(path)) # Explicit path for plotting     

    pickle.dump(path, open("path.dat","wb"))

    path = on_path(path, rsc, irsc) # Points of the reciprocal lattice on the path
    #matdyn.path = closest_box(path, rsc, irsc) # Closest 8 points to reciprocal lattice
    path.extend(all_points(rpc, irpc, rsc, irsc))
    path = unique(np.array(path)).tolist()
    matdyn.path = derivative_points(path, rsc) # All the points in the SC reciprocal space that are inside the PC brilliouin zone
    

    apath = np.array(matdyn.path) # Path in array for for plotting 
    
    # Displaying the high symmetry path
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(apath[:,0], apath[:,1], apath[:,2])
    ax.plot(epath[:,0], epath[:,1], epath[:,2])
    rpc = rpc / (2*np.pi)
    ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), rpc[:,0], rpc[:,1], rpc[:,2])
    
    ax.view_init(-45,-45)
    plt.savefig('Qpath.png')    
    
    # Submitting all the jobs
    # submit_jobs(pwscf, ph, q2r, matdyn, np = nproc)
    submit_jobs(matdyn, np = 1)
    
    pickle.dump(matdyn.read_eig(), open("eigpc.dat","wb"))

    # Super Cell Calculation #############################################################

    # ------> No relaxation for testing 
    # Relaxation
    pwrelax = pwcalc()
    pwrelax.name = "sc"
    pwrelax.calc_type = "relax"
    pwrelax.restart_mode = "from_scratch"
    pwrelax.pseudo_dir = os.path.expanduser("~/scratch/pseudo_pz-bhs/")
    pwrelax.celldm = 10.7
    pwrelax.ecutwfc = 45.0
    pwrelax.ecutrho = 400.0
    pwrelax.nbnd = len(perfectStrucsc)*4
    pwrelax.occupations = "fixed"
    pwrelax.masses = {'Si':pt.Si.atomic_weight}
    pwrelax.from_pylada(perfectStrucsc)
    pwrelax.kpoints = [1,1,1]
    
    # #submit_jobs(pwrelax, np = nproc)
    # ene = pwrelax.read_energies()
    # while (abs(ene[-1] - ene[-2]) > 1e-8):
    #     pwrelax.atomic_pos = pwrelax.read_atomic_pos()
    #     pwrelax.cell = pwrelax.read_cell()
    #     submit_jobs(pwrelax, np = nproc)
    #     ene = pwrelax.read_energies()
    
    # Self consistant run
    pwscf = deepcopy(pwrelax)
    pwscf.calc_type = 'scf'
    pwscf.from_pylada(Strucsc)
    # pwscf.atomic_pos = pwrelax.read_atomic_pos() # With relaxation
    # pwscf.cell = pwrelax.read_cell() # With relaxation

    # Phonons
    ph = phcalc()
    
    ph.name = pwscf.name
    ph.masses = pwscf.masses
    ph.qpoints = [2,2,2]
    # ph.ldisp = False
    # ph.qlist = [[0.0,0.0,0.0]]    

    # dynmat = dyncalc()
    # dynmat.name = pwscf.name
    
    # Inverse Fourier transform
    q2r = q2rcalc()
    
    q2r.name = pwscf.name
    
    # Fourier transform
    matdyn = matcalc()
    
    matdyn.name = pwscf.name
    matdyn.masses = pwscf.masses

    matdyn.path = [[0,0,0,1]]

    submit_jobs(pwscf, ph, np = nproc)
    submit_jobs(q2r, matdyn, np = 1)

    pickle.dump(matdyn.read_eig(), open("eigsc.dat","wb"))

