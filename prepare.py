from pythonQE import pwcalc, phcalc, q2rcalc, matcalc, submit_jobs
from copy import deepcopy
import os
from pylada.crystal import supercell, Structure
import pylada.periodic_table as pt
import pickle
import numpy as np

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
        return np.ceil(vec)
    else:
        return np.floor(vec)

def closest_box(epathrc):
    closest = [] 
    for i in range(2):
        for j in range(2):
            for k in range(2):
                closest.append(np.concatenate((approx(i,epathrc[:,0:1]), approx(j,epathrc[:,1:2]), approx(k,epathrc[:,2:3])), axis=1))

    return  np.concatenate(closest, axis=0)

           

# Primittive structure
A = Structure([[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5]])
A.add_atom(0,0,0,'Si')
A.add_atom(0.25,0.25,0.25,'Si')

# Building the (perfect) supercell
Asc = supercell(A,[[10,0,0],[0,10,0],[0,0,10]]);


# # Super Cell Calculation #############################################################

# pwrelax = pwcalc()
# pwrelax.name = "sc"
# pwrelax.calc_type = "vc-relax"
# pwrelax.restart_mode = "recover"
# pwrelax.pseudo_dir = os.path.expanduser("~/scratch/pseudo_pz-bhs/")
# pwrelax.celldm = 10.7
# pwrelax.ecutwfc = 45.0
# pwrelax.ecutrho = 400.0
# pwrelax.nbnd = len(Asc)*4
# pwrelax.occupations = "fixed"
# pwrelax.masses = {'Si':pt.Si.atomic_weight}
# pwrelax.from_pylada(Asc)
# pwrelax.kpoints = [8,8,8]

# pwrelax.write_in()

# submit_jobs(pwrelax, np = 48)
# ene = pwrelax.read_energies()
# while (abs(ene[-1] - ene[-2]) > 1e-8):
#     pwrelax.atomic_pos = pwrelax.read_atomic_pos()
#     pwrelax.cell = pwrelax.read_cell()
#     submit_jobs(pwrelax, np = 48)
#     ene = pwrelax.read_energies()
# pwscf = deepcopy(pwrelax)
# pwscf.calc_type = 'scf'
# pwscf.atomic_pos = pwrelax.read_atomic_pos()
# pwscf.cell = pwrelax.read_cell()

# ph = phcalc()

# ph.name = pwscf.name
# ph.masses = pwscf.masses
# ph.qpoints = [0,0,0]

# q2r = q2rcalc()

# q2r.name = pwscf.name

# matdyn = matcalc()

# matdyn.name = pwscf.name
# matdyn.masses = pwscf.masses
# matdyn.path = [
#     [0.0000000,   0.0000000,   0.0000000, 1]]

# submit_jobs(pwscf, ph, q2r, matdyn, np = 48)

# pickle.dump(matdyn.read_eig(), open("eigsc.dat","wb"))

# # Primitive Cell Calculations ########################################################

# pwrelax = pwcalc()
# pwrelax.name = "pc"
# pwrelax.calc_type = "vc-relax"
# pwrelax.restart_mode = "from_scratch"
# pwrelax.pseudo_dir = os.path.expanduser("~/scratch/pseudo_pz-bhs/")
# pwrelax.celldm = 10.7
# pwrelax.ecutwfc = 45.0
# pwrelax.ecutrho = 400.0
# pwrelax.nbnd = len(A)*4
# pwrelax.occupations = "fixed"
# pwrelax.masses = {'Si':pt.Si.atomic_weight}
# pwrelax.from_pylada(A)
# pwrelax.kpoints = [8,8,8]

# pwrelax.write_in()

# submit_jobs(pwrelax, np = 48)
# ene = pwrelax.read_energies()
# while (abs(ene[-1] - ene[-2]) > 1e-8):
#     pwrelax.atomic_pos = pwrelax.read_atomic_pos()
#     pwrelax.cell = pwrelax.read_cell()
#     submit_jobs(pwrelax, np = 48)
#     ene = pwrelax.read_energies()
# pwscf = deepcopy(pwrelax)
# pwscf.calc_type = 'scf'
# pwscf.atomic_pos = pwrelax.read_atomic_pos()
# pwscf.cell = pwrelax.read_cell()

# ph = phcalc()

# ph.name = pwscf.name
# ph.masses = pwscf.masses
# ph.qpoints = [6,6,6]

# q2r = q2rcalc()

# q2r.name = pwscf.name

# matdyn = matcalc()

# matdyn.name = pwscf.name
# matdyn.masses = pwscf.masses

rsc = reciprocal(Asc.cell)
irsc = np.linalg.inv(rsc)

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


# submit_jobs(pwscf, ph, q2r, matdyn, np = 48)

# pickle.dump(matdyn.read_eig(), open("eigpc.dat","wb"))

explicit_path = np.array(explicit_path(path))*np.pi*2
explicit_path_reciprocal = explicit_path[:,0:3].dot(irsc)

box_path = closest_box(explicit_path_reciprocal)

path = unique(box_path).dot(rsc)
path = np.concatenate((path, np.ones((np.shape(path)[0],1))), axis=1) #Adding ones for QE

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(path[:,0], path[:,1], path[:,2])
ax.plot(explicit_path[:,0], explicit_path[:,1], explicit_path[:,2])


ax.view_init(-30, -60)
plt.savefig('test.png')

plt.show()

