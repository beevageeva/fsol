from scipy.fftpack import fft2,fftfreq,fftshift #forFourierTransform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import pi


a = 0
b = 4
numPoints = 1024
B0 = 10
xx = np.linspace(a,b, numPoints)
x,y =  np.meshgrid(xx, xx, indexing='ij')
Bz0 = B0 * np.cos(4 * pi * (x-a)/(b-a) + 2 * pi * (y-a)/(b-a))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot_surface(x,y,Bz0)
		
Y=fft2(Bz0)

F = fftfreq(numPoints, x[1] - x[0])
eps = 1
ind = np.where(np.absolute(Y) - eps>0)
print(ind[0].shape)

print("freqs")
print(F)
print("coefs")
print(Y)



		
plt.draw()
plt.show()


##The mean is Y[0,0]
#Y[0,0] = 0
