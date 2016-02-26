import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

z0 = 0
zf = 100
x0 = 0
xf = 10
dt = 0.1
timeEnd = 100
numPointsX = 128
numPointsZ = 128
omega = 1
A = 1


z,x =  np.meshgrid(np.linspace(z0,zf, numPointsZ), np.linspace(x0,xf, numPointsX), indexing='ij')

va1 = 2
va2 = 3

#tanh
#xe = 0.5*(xf + x0) #middle
#we = 0.5
##we = 0.4
##we = 0.2
#va = va1 + 0.5 * (va2-va1) * (1 + np.tanh((x-xe)/we))

#line
va = va1 + ((va2 - va1) * (x-x0)) / (xf - x0)


t = 0
plt.xlabel("z")
plt.ylabel("x")
while t < timeEnd:
	vals = np.cos(omega*(z/va - t))
	valsz = vals[0.5 * (zf + z0), :]
	#valsz = vals[:,0.5 * (zf + z0)]
	plt.cla()
	plt.title("Time %.1f" % t)
	plt.plot(x, valsz)
	#plt.imshow(vals)
	plt.draw()
	plt.show(block=False)
	t+=dt
