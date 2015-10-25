import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



def plotTg(M):
	plt.title("tan x = x, x in [0,%d pi/2]" % (2*M-1))
	numPoints = 500
	from math import pi
	dx= 0.01
	x = np.linspace(0, 0.5 * pi-dx, numPoints, endpoint=True)
	plt.plot(x, np.tan(x), "r-", label="y = tan x")
	plt.plot(x, x, "g-", label="y = x")
	
	for i in np.arange(1,2*M-1 ,2):
		#x = np.concatenate((x, np.linspace(i*pi+2 * dx, (i+1)*pi-2*dx, 2 * numPoints, endpoint=True)), axis=0)
		x = np.linspace(0.5 * i*pi+2 * dx, 0.5 * (i+2)*pi-2*dx, 2 * numPoints, endpoint=True)
		plt.plot(x, np.tan(x), "r-", label="y = tan x")
		plt.plot(x, x, "g-", label="y = x")
		plt.vlines(0.5 * i * pi ,-M *pi,M*pi,color='k',linestyles='solid')
	plt.vlines(0.5 * (2*M -1) * pi ,-M *pi,M*pi,color='k',linestyles='solid')
	plt.ylim((-M * pi , M * pi))
	plt.xlim((0 , 0.5 * (2 * M - 1)* pi))
	#plt.legend()
	plt.draw()
	plt.show()


plotTg(5)

