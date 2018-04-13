import numpy as np
from pylada.crystal import supercell, Structure
from prepare import reciprocal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

tol = 1e-12

# Primittive structure
A = Structure([[0.5, 0.5, 0],[0.5, 0, 0.5],[0, 0.5, 0.5]])
A.add_atom(0,0,0,'Si')
A.add_atom(0.25,0.25,0.25,'Si')

# Building the (perfect) supercell
Asc = supercell(A,[[3,0,0],[0,3,0],[0,0,3]]);

rpc = reciprocal(A.cell) #reciprocal lattice of PC
irpc = np.linalg.inv(rpc) #inverse of reciprocal lattice of PC

rsc = reciprocal(Asc.cell) #reciprocal lattice of SC
irsc = np.linalg.inv(rsc) #inverse of reciprocal lattice of SC

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

list_in = np.array(list_in)

# Uniform grid

x = np.linspace(-0.5,0.5,7)[:-1]

qgrid = np.array(np.meshgrid(x,x,x)).reshape(3,-1,order = 'F').T.dot(rpc)

# Easy PC
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(list_in[:,0], list_in[:,1], list_in[:,2])
ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), rpc[:,0], rpc[:,1], rpc[:,2])
ax.quiver(np.zeros(3), np.zeros(3), np.zeros(3), rsc[:,0], rsc[:,1], rsc[:,2], color ='red')

qpoints = np.array([
        [0.000000000,   0.000000000,   0.000000000],
        [0.167593549,   0.167593549,   0.167593549],
        [0.335187099,   0.335187099,   0.335187099],
        [0.502780648,  -0.502780648,  -0.502780648],
        [0.000000000,   0.000000000,   0.335187099],
        [0.167593549,   0.167593549,   0.502780648],
        [0.670374198,  -0.670374198,  -0.335187099],
        [0.502780648,  -0.502780648,  -0.167593549],
        [0.335187099,  -0.335187099,   0.000000000],
        [0.000000000,   0.000000000,   0.670374198],
        [0.837967747,  -0.837967747,  -0.167593549],
        [0.670374198,  -0.670374198,  -0.000000000],
        [0.000000000,   0.000000000,  -1.005561297],
        [1.005561297,  -0.670374198,  -0.335187099],
        [0.837967747,  -0.502780648,  -0.167593549],
        [0.000000000,   0.335187099,  -1.005561297]])*2*np.pi

ax.scatter(qpoints[:,0], qpoints[:,1], qpoints[:,2], color = 'green')
ax.scatter(qgrid[:,0], qgrid[:,1], qgrid[:,2], color = 'orange')

ax.view_init(40,-45)
plt.savefig('In_PC_6x6_40.45.png')

print np.shape(list_in)

plt.show()
