from pythonQE import *
from copy import deepcopy
import os
from pylada.crystal import supercell, Structure
import pylada.periodic_table as pt
import pickle
import numpy as np

nproc = 96

# Primittive structure
perfectStruc = Structure([[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5]])
perfectStruc.add_atom(0,0,0,'Si')
perfectStruc.add_atom(0.25,0.25,0.25,'Si')

# Building the (perfect) supercell
perfectStrucsc = supercell(perfectStruc,[[2,0,0],[0,2,0],[0,0,2]]);
    
# Primitive Cell Calculations ########################################################

# Relaxation
pwrelax = pwcalc()
pwrelax.name = "pc666"
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
pwrelax.kpoints = [6,6,6]

submit_jobs(pwrelax, np = nproc)
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
print "length of path", len(matdyn.path)

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
submit_jobs(pwscf, ph, np = nproc)
submit_jobs(q2r, matdyn, np = 1)

pickle.dump(matdyn.read_eig(), open("eigpc.dat","wb"))

# Super Cell Calculation #############################################################

# ------> No relaxation for testing 
# Relaxation
pwrelax = pwcalc()
pwrelax.name = "222sc"
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
pwrelax.kpoints = [2,2,2]

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
pwscf.unit = "crystal"
pwscf.from_pylada(Strucsc)

# pwscf.atomic_pos = pwrelax.read_atomic_pos() # With relaxation
# pwscf.cell = pwrelax.read_cell() # With relaxation

# Phonons
ph = phcalc()

ph.name = pwscf.name
ph.masses = pwscf.masses
# ph.qpoints = [2,2,2]
ph.ldisp = False
ph.qlist = [[0.0,0.0,0.0]]    

dynmat = dyncalc()
dynmat.name = pwscf.name

# Inverse Fourier transform
# q2r = q2rcalc()

# q2r.name = pwscf.name

# Fourier transform
# matdyn = matcalc()

# matdyn.name = pwscf.name
# matdyn.masses = pwscf.masses

# matdyn.path = [[0,0,0,1]]

# Submit jobs
submit_jobs(pwscf, pw, np = nproc)
# submit_jobs(q2r, matdyn, np = 1)
submit_jobs(dynmat, np = 1)

pickle.dump(dynmat.read_eig(), open("eigsc.dat","wb"))

