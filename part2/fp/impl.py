from scipy.fftpack import fft2,fftfreq,fftshift #forFourierTransform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import pi,e,sqrt

Lx = 5
Ly = 8

numPoints = 1024
xx = np.linspace(-Lx,Lx, numPoints)
yy = np.linspace(0,Ly, numPoints)
x,y =  np.meshgrid(xx, yy, indexing='ij')
Bz0 = sqrt(2)
Bp0 = Bz0/sqrt(1-1/e**3)

Phi = 10/pi * Bp0 * np.cos(pi * x / 10)*np.exp(-pi * y / 10) - 10/(3*pi) * Bp0 * np.cos(3*pi*x/10) * np.exp(-3*pi*y/10)

Bx =  Bp0 * np.cos(pi * x / 10)*np.exp(-pi * y / 10) - Bp0 * np.cos(3*pi*x/10) * np.exp(-3*pi*y/10)
By = -Bp0 * np.sin(pi * x / 10)*np.exp(-pi * y / 10) + Bp0 * np.sin(3*pi*x/10) * np.exp(-3*pi*y/10)

fig = plt.figure()
def plotImpl(proj = False):
	if proj:
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.plot_surface(x,y,Phi)
	else:
		ax = fig.add_subplot(111)
		ax.set_xlabel("y")
		ax.set_ylabel("x")
		ax.imshow(Phi)

def plotStr():		
	ax = fig.add_subplot(111)
	ax.set_xlabel("y")
	ax.set_ylabel("x")
	magB = np.sqrt(Bx**2+By**2)
	normMag = magB / magB.max()
	ax.streamplot(y,x,Bx,By, color=normMag, linewidth=normMag*5)

def plotMag():
	ax = fig.add_subplot(111)
	ax.set_xlabel("y")
	ax.set_ylabel("x")
	ax.imshow(np.sqrt(Bx**2,By**2))


def plotHeating(proj=False):
	sy = 0.25
	sx = 0.2
	yh = 0.4
	xl = -4.2
	xr = 4.2
	heatbg = 1.5e-2 * np.exp(-y/5)	
	cy = np.zeros(y.shape)
	for i in range(y.shape[0]):
		for j in range(y.shape[1]):
			if(y[i,j]<yh):	
				cy[i,j] = 1
			else:
				cy[i,j] = np.exp(-(y[i,j]-yh)**2/sy)
	heat = cy * (np.exp(-(x-xl)**2/sx) + np.exp(-(x-xr)**2/sx) )
	#heat = (np.exp(-(x-xl)**2/sx) + np.exp(-(x-xr)**2/sx) )
	if proj:
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		#ax.plot_surface(x,y,cy)
		#ax.plot_surface(x,y,heat)
		ax.plot_surface(x,y,heat + heatbg)
	else:
		ax = fig.add_subplot(111)
		ax.set_xlabel("y")
		ax.set_ylabel("x")
		#ax.imshow(heat)
		ax.imshow(heatbg + heat)

#plotImpl(True)
#plotStr()
#plotMag()
plotHeating(True)
		
plt.draw()
plt.show()

