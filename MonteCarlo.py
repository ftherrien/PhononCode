import argparse
import numpy as np
import time
import PhononCode as pc
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

# Initial MPI calls
comm = MPI.COMM_WORLD
master = 0
n_proc = comm.Get_size()
rank = comm.Get_rank()

if rank==master:
    totalTime = time.time()
    iterTime = time.time()
    
parser = argparse.ArgumentParser()
parser.add_argument("-T","--Ttol",dest="Ttol",type=float, default=1.5, help="Number of iteration")
parser.add_argument("-n","--numIter",dest="numIter",type=int, default=4000, help="Number of iterations")
parser.add_argument("-s","--size",dest="n",type=int, default=300, help="Number of k points")
parser.add_argument("-o","--output",dest="folder",type=str, default='test/', help="Folder name")
parser.add_argument("-k","--bond",dest="ke",type=float, default='test/', help="Bond strength")
parser.add_argument("-b","--interbond",dest="ki",type=float, default='test/', help="Bond strenght between defects")
parser.add_argument("-m","--mass",dest="m",type=float, default='test/', help="Defect mass")
parser.add_argument("-c","--defConc",dest="defConc",type=float, default=0.3, help="Defect concentration")
parser.add_argument("-i","--init",dest="init",type=str, default='ordered', help="Initial structure")

options = parser.parse_args()



# Fixed PhononCode params
n = options.n
nb = 1
nE =150
T = 300
V = np.array([1])*(1+0*1j)
Mvec = np.array([1])
defects = True
DefConc = [options.defConc]
dtype = ["custom"]
mval = np.array([[options.m]])
kval = [[np.array([[ options.ke,  options.ke], [ options.ki,  options.ki]]), []]]
clusterSize = [1]
gaussian = True
w = 0.004
repeat=1
avg=False
CutOffErr = 1e-4
folder = 'StatMecProj/' + options.folder
show = False

args = (n, nb, nE, T, V, Mvec,
        defects, DefConc, dtype, mval, kval, clusterSize,
        gaussian, w, repeat, avg, CutOffErr,
        folder, show)

# Interresting comparison
nDefect = int(DefConc[0]*n)
layout = np.zeros(n)
pos = np.arange(0, n, int(1 / DefConc[0]))
layout[pos] = 1
layord = layout

LTC_ordered = pc.PhononCode(*(args+(layout,False)))

# pos = np.array([],np.int)
# for i in range(4):
#     pos = np.append(pos,np.arange(i*10,300,50,np.int))
# pos = np.append(pos,np.arange(4*10+1,300,50,np.int))
# layout = np.zeros(300)
# layout[pos] = 1
# laysord = layout

# LTC_sub_ordered = pc.PhononCode(*(args+(layout,False)))

layout = np.zeros(n)
pos = np.arange(0, nDefect)
layout[pos] = 1
layclu = layout

LTC_clustered = pc.PhononCode(*(args+(layout,False)))

if rank == master:
    print('LTC Ordered:', LTC_ordered)
    # print('LTC Sub Ordered:', LTC_sub_ordered)
    print('LTC Clustered:', LTC_clustered)

# Other Params
numIter = options.numIter
Ttol = options.Ttol

# Initial layout
if options.init == 'ordered':
    layout = layord
# elif options.init == 'subordered':
#     layout = laysord
elif options.init == 'clustered':
    layout = layclu
else: 
    layout = np.zeros(n)
    layout[np.random.choice(np.arange(0,n), nDefect, replace=False)] = 1

nDefect = np.sum(layout)    

# Initialization
LTCplot = np.ones(numIter)*np.Inf
LTCmin = np.Inf
layouts = []

for i in range(numIter):

    LTC = pc.PhononCode(*(args+(layout,False)))

    if rank == master:
    
        if LTC < LTCmin:
            LTCmin = LTC
            bestLayout = layout
        elif np.random.random() <= np.exp((LTCmin - LTC)/Ttol):
            LTCmin = LTC
            bestLayout = layout

        layout = np.array(bestLayout)
        while (layout == bestLayout).all():
            # Find the position of a random defect
            pos = np.nonzero(bestLayout == 1)[0][np.random.randint(nDefect)]
            one = np.random.randint(2)-1
            layout[pos] = bestLayout[(pos+one)%n]
            layout[(pos+one)%n] = bestLayout[pos]
        
        LTCplot[i] = LTCmin
        layouts.append(bestLayout)
        
        if i%10 == 0:
            print('Iteration: %d, LTC: %f, Elapsed time: %f'%(i,LTCmin,time.time()-iterTime))
            print('Best LTC so far:', LTCplot[np.argmin(LTCplot)])
            print('Best layout so far:', layouts[np.argmin(LTCplot)])
            iterTime = time.time()

if rank == master:
    plt.plot(LTCplot)
    plt.xlabel('Iteration number')
    plt.ylabel('Optimal LTC')
    plt.title("Optimisation of the LTC")
    plt.savefig(folder+"optimisation.png")

    print('Best laytout:', layouts[np.argmin(LTCplot)])
    print('Lowest LTC:', LTCplot[np.argmin(LTCplot)])
    totalTime = time.time() - totalTime
    print('Total optimisation time: %f'%totalTime)
    layout = layouts[np.argmin(LTCplot)]

LTC_opt = pc.PhononCode(*(args+(layout,True)))

if rank == master:
    print('LTC Optimal:', LTC_opt)
    print('==================================')
