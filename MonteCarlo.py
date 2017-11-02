import numpy as np
import time
import PhononCode as pc
from mpi4py import MPI
import matplotlib.pyplot as plt

# Initial MPI calls
comm = MPI.COMM_WORLD
master = 0
n_proc = comm.Get_size()
rank = comm.Get_rank()

# Fixed PhononCode params
n = 40
nb = 1
nE = 20
T = 300
V = np.array([1])*(1+0*1j)
Mvec = np.array([1])
defects = True
DefConc = [0.5]
dtype = ["custom"]
mval = np.array([[2]])
kval = [[np.array([[ 1,  1], [ 1,  1]]), []]]
clusterSize = [1]
gaussian = True
w = 1/n
repeat=1
avg=False
CutOffErr = 1e-4
folder = 'StatMecProj'
show = True

args = (n, nb, nE, T, V, Mvec,
        defects, DefConc, dtype, mval, kval, clusterSize,
        gaussian, w, repeat, avg, CutOffErr,
        folder, show)

# Other Params
numIter = 1000
Ttol = 0.7

# Initial layout
nDefect = int(DefConc[0]*n)
layout = np.zeros(n)
layout[np.random.choice(np.arange(0,n), nDefect, replace=False)] = 1
    
# Initialization
LTCplot = np.zeros(numIter)
LTCmin = np.Inf
layouts = []

for i in range(numIter):

    LTC = pc.PhononCode(*args,layout,False)

    if rank == master:
    
        if LTC < LTCmin:
            LTCmin = LTC
            bestLayout = layout
        elif np.random.random() <= np.exp((LTCmin - LTC)/Ttol):
            LTCmin = LTC
            bestLayout = layout
        
        # Find the position of a random defect
        pos = np.nonzero(bestLayout == 1)[0][np.random.randint(nDefect)]
        
        layout = np.array(bestLayout)
        layout[pos] = bestLayout[(pos+1)%n]
        layout[(pos+1)%n] = bestLayout[pos]
        
        LTCplot[i] = LTCmin
        layouts.append(bestLayout)
        
        if i%100 == 0:
            print('Iteration: %d, LTC: %f'%(i,LTCmin))

if rank == master:
    plt.plot(LTCplot)
    print(layouts[np.argmin(LTCplot)])
    print(LTCplot[np.argmin(LTCplot)])
    layout = layouts[np.argmin(LTCplot)]
    plt.show()
    
LTC = pc.PhononCode(*args,layout,True)

LTC = pc.PhononCode(n, nb, nE, T, V, Mvec,
        defects, DefConc, ["ordered"], mval, kval, clusterSize,
        gaussian, w, repeat, avg, CutOffErr,
        folder, show)
