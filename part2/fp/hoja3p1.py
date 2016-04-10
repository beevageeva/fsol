from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import pi,e,sqrt


Lx = 1
Lz = 1
alpha = 0.1
B0 = 1.0

numPoints = 1024
xx = np.linspace(0,Lx, numPoints)
zz = np.linspace(0,Lz, numPoints)
x,z =  np.meshgrid(xx, zz, indexing='ij')

Bx = B0 * alpha * (x+z)
Bz = B0 +  B0 * alpha * (x-z)


plt.xlabel("z")
plt.ylabel("x")
magB = np.sqrt(Bx**2+Bz**2)
normMag = magB / magB.max()
plt.streamplot(z,x,Bx,Bz, color=normMag, linewidth=normMag*5)

plt.draw()
plt.show()


