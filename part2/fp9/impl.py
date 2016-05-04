from scipy.fftpack import fft2,fftfreq,fftshift #forFourierTransform
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from math import pi,e,sqrt

Lx = 10
Ly = 10
B0 = 1.0

numPoints = 1024
xx = np.linspace(-Lx,Lx, numPoints)
yy = np.linspace(-Ly,Ly, numPoints)
x,y =  np.meshgrid(xx, yy, indexing='ij')
Bx = B0 + np.zeros((numPoints,numPoints))
By = 2 * B0 * x



fig = plt.figure()

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



plotStr()
#plotMag()
		
plt.draw()
plt.show()

