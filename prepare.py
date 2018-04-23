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
                linpath = np.reshape(np.linspace(line[j],path[i+1][j],line[-1]+1), (1,line[-1]+1))
            else:
                linpath = np.concatenate((linpath, np.reshape(np.linspace(line[j],path[i+1][j],line[-1]+1), (1,line[-1]+1))), axis=0)
        linpath = np.concatenate((linpath, np.ones((1,line[-1]+1))),axis=0)
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
    
    path = np.array(path)[:,:3]
    
    closest_points = unique(np.round(epathrc)).dot(rsc) / (2*np.pi)
    onpath = []
    for i,v in enumerate(path[:-1,:]):
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
         

def on_path_non_unique(path, rsc, irsc):
    epath = np.array(explicit_path(path))*np.pi*2
    epathrc = epath[:,0:3].dot(irsc)
    
    npts = np.array(path)[:,3]
    path = np.array(path)[:,:3]
    
    closest_points = unique(np.round(epathrc)).dot(rsc) / (2*np.pi)
    onpath = []
    dist_on_path = 0
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
            dist_on_path =+ np.linalg.norm(path[i+1]-v)
                 
    newpath = np.array(onpath) 
         
    return (np.concatenate((newpath, np.ones((np.shape(newpath)[0],1))), axis=1).tolist(),
            pos_on_path, dist_on_path) #Adding ones for QE

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
                if (p.dot(irpc) >= -0.5).all() and (p.dot(irpc) < 0.5 - tol).all():
                    list_in.append(p.tolist())
    
    list_in = np.array(list_in) / (2*np.pi)

    return np.concatenate((list_in, np.ones((np.shape(list_in)[0],1))), axis=1).tolist() #Adding ones for QE


# Begin program ======================================================================
if __name__ == "__main__":
    nproc = 96
    
    # Primittive structure
    A = Structure([[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5]])
    A.add_atom(0,0,0,'Si')
    A.add_atom(0.25,0.25,0.25,'Si')
    
    # Building the (perfect) supercell
    Asc = supercell(A,[[3,0,0],[0,3,0],[0,0,3]]);
    
    pickle.dump((A, Asc), open("structures.dat","wb"))

    # Super Cell Calculation #############################################################
    
    # Relaxation
    pwrelax = pwcalc()
    pwrelax.name = "sc"
    pwrelax.calc_type = "vc-relax"
    pwrelax.restart_mode = "from_scratch"
    pwrelax.pseudo_dir = os.path.expanduser("~/scratch/pseudo_pz-bhs/")
    pwrelax.celldm = 10.7
    pwrelax.ecutwfc = 45.0
    pwrelax.ecutrho = 400.0
    pwrelax.nbnd = len(Asc)*4
    pwrelax.occupations = "fixed"
    pwrelax.masses = {'Si':pt.Si.atomic_weight}
    pwrelax.from_pylada(Asc)
    pwrelax.kpoints = [1,1,1]
    
    pwrelax.write_in()    
    #submit_jobs(pwrelax, np = nproc)
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
    ph.ldisp = False
    ph.qlist = [[0.0,0.0,0.0]]    
       
    # Fourier Transform
    dynmat = dyncalc()
    
    dynmat.name = pwscf.name
    
    #submit_jobs(pwscf, ph, np = nproc)
    #submit_jobs(dynmat, np = 1)

    pickle.dump(dynmat.read_eig(), open("eigsc.dat","wb"))
    
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
    pwrelax.nbnd = len(A)*4
    pwrelax.occupations = "fixed"
    pwrelax.masses = {'Si':pt.Si.atomic_weight}
    pwrelax.from_pylada(A)
    pwrelax.kpoints = [8,8,8]
    
    pwrelax.write_in()
    
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
    
    rsc = reciprocal(Asc.cell) #reciprocal lattice
    irsc = np.linalg.inv(rsc) #inverse of reciprocal lattice

    rpc = reciprocal(A.cell) #reciprocal lattice
    irpc = np.linalg.inv(rpc) #inverse of reciprocal lattice
    
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

    epath = np.array(explicit_path(path)) # Explicit path for plotting     

    pickle.dump(path, open("path.dat","wb"))

    path = on_path(path, rsc, irsc) # Points if the reciprocal lattice on the path
    #matdyn.path = closest_box(path, rsc, irsc) # Closest 8 points to reciprocal lattice
    path.extend(all_points(rpc, irpc, rsc, irsc))
    matdyn.path = unique(np.array(path)).tolist() # All the points in the SC reciprocal space that are inside the PC brilliouin zone
    
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
    
    ax.view_init(0,180)
    plt.savefig('Qpath.png')    
    
    # Submitting all the jobs
    # submit_jobs(pwscf, ph, q2r, matdyn, np = nproc)
    submit_jobs(matdyn, np = 1)
    
    pickle.dump(matdyn.read_eig(), open("eigpc.dat","wb"))    
