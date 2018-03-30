from pythonQE import pwcalc, phcalc, q2rcalc, matcalc, submit_jobs
from copy import deepcopy
import os
from pylada.crystal import supercell, Structure
import pylada.periodic_table as pt
import numpy as np

# Primittive structure
A = Structure([[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5]])
A.add_atom(0,0,0,'Si')
A.add_atom(0.25,0.25,0.25,'Si')

# Building the perfect supercell
Asc = supercell(A,[[3,0,0],[0,3,0],[0,0,3]]);

pwrelax = pwcalc()
pwrelax.name = "sc"
pwrelax.calc_type = "vc-relax"
pwrelax.restart_mode = "from_scratch"
pwrelax.pseudo_dir = os.path.expanduser("~/scratch/pseudo_pz-bhs/")
pwrelax.celldm = 10.7
pwrelax.ecutwfc = 45.0
pwrelax.ecutrho = 400.0
pwrelax.nbnd = 28
pwrelax.occupations = "fixed"
pwrelax.masses = {'Si':pt.Si.atomic_weight}
pwrelax.from_pylada(Asc)
pwrelax.kpoints = [10,10,10]

pwrelax.write_in()

submit_jobs(pwrelax, np = 48)
ene = pwrelax.read_energies()
while (abs(ene[-1] - ene[-2]) > 1e-8):
    pwrelax.atomic_pos = pwrelax.read_atomic_pos()
    pwrelax.cell = pwrelax.read_cell()
    submit_jobs(pwrelax, np = 48)
    ene = pwrelax.read_energies()
pwscf = deepcopy(pwrelax)
pwscf.calc_type = 'scf'
pwscf.atomic_pos = pwrelax.read_atomic_pos()
pwscf.cell = pwrelax.read_cell()

ph = phcalc()

ph.name = pwscf.name
ph.masses = pwscf.masses
ph.qpoints = [10,10,10]

q2r = q2rcalc()

q2r.name = pwscf.name

matdyn = matcalc()

matdyn.name = pwscf.name
matdyn.masses = pwscf.masses
matdyn.path = [
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

submit_jobs(pwscf, ph, q2r, matdyn, np = 48)

print matdyn.read_eig()
