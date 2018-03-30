import numpy as np
from pythonQE import matcalc
from mpi4py import MPI

# Initial MPI calls                                                                   
comm = MPI.COMM_WORLD
master = 0
n_proc = comm.Get_size()
rank = comm.Get_rank()

matdyn = matcalc()
matdyn.name = "si"

qsc, omegasc, eigenvecsc = matdyn.read_eig()
