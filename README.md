# PhononCode

Calculates Lattice Thermal Conductivity and band structure of an arbitrary 1D chain of atoms with arbitrary defect type, concentration and distribution.

The PHONON.out file contains one line information about the run. It will not be overwritten by each run, it is meant to be able to plot the results.

The PARAM.out contains all the parameters of the run that created its parent folder.

The code is parallelizable up to n the number of primittive cell in the super cell.

Specific instructions:

Bond strenght (k) is specified with a list of form: [[e_{0l},e_{0r}],...,[e_{(nd-1)l},e_{(nd-1)r}],e_nd,[i_1,i_2,...,i_{nb-1}]]
    Where:
    e_{0l} is the interaction on the 'left' (l) side of the cell with the primitive cell (0)
    e_{1l} is the interaction on the 'left' (l) side of the cell with the first type of defect
    i_1 is the first internal bond in the defect

Requires: numpy, matplotlib, mpi4py, scipy


Usage: PhononCode.py [-h] [-b NB] [-n N] [-E NE] [-T T] [-V V] [-M MVEC] [-g]
                     [-u CUTOFFERR] [-w W] [-d] [-t DTYPE [DTYPE ...]]
                     [-s CLUSTERSIZE [CLUSTERSIZE ...]] [-m MVAL [MVAL ...]]
                     [-v KVAL [KVAL ...]] [-c DEFCONC [DEFCONC ...]]
                     [-o FOLDER] [-i]

optional arguments:
  -h, --help            show this help message and exit
  -b NB, --pcsize NB    Size of primittive cell
  -n N, --scsize N      Number of primittive cell in super cell
  -E NE, --energy NE    Number of energy values on scale
  -T T, --temp T        Temperature
  -V V, --potentials V  List of potential strenghts of size nb
  -M MVEC, --masses MVEC
                        List of masses of size nb
  -g, --noGauss         Energy bands with minimum gaussian width
  -u CUTOFFERR, --cutOffErr CUTOFFERR
                        Value at witch a band is considered to be negligable
  -w W, --width W       Standard deviation (width) of gaussian
  -d, --defects         Adds defects in super cell
  -t DTYPE [DTYPE ...], --dtype DTYPE [DTYPE ...]
                        Available types: ordered, random, cluster
  -s CLUSTERSIZE [CLUSTERSIZE ...], --cluster CLUSTERSIZE [CLUSTERSIZE ...]
                        Size factor for clustered defects
  -m MVAL [MVAL ...], --dmasses MVAL [MVAL ...]
                        Mass of defects
  -v KVAL [KVAL ...], --dpotentials KVAL [KVAL ...]
                        Potential strenghts of defects
  -c DEFCONC [DEFCONC ...], --defConc DEFCONC [DEFCONC ...]
                        Concentration of defect
  -o FOLDER, --out FOLDER
                        Output folder for images
  -i, --display         Display generated plots plots

Examples:

Serial call:
python3 PhononCode.py -b 2 -V [1,1] -M [1,1] -n 50 -d -t ordered random -v [[2.1,2.1],2.2,2] [[3.1,3.1],[3.2,3.2],3.3,3] -m [2,2] [3,3] -s 3 3 -c 0.3 0.3 -i


Parallel call:
mpiexec -n 4 python3 PhononCode.py -b 2 -V [1,1] -M [1,1] -n 50 -d -t ordered random -v [[2.1,2.1],2.2,2] [[3.1,3.1],[3.2,3.2],3.3,3] -m [2,2] [3,3] -s 3 3 -c 0.3 0.3 -o run1




