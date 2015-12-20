import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

z0 = 0
zf = 4e7 #40Mm
numPointsZ = 1024




		from scipy.fftpack import fft2,fftfreq,fftshift #forFourierTransform
		
		##vals = [intlen * abs(Y), F * intlen]	
		#vals = [intlen * abs(Y), F / (2 * pi)]
		nx = vals.shape[0]	
		ny = vals.shape[1]	
		numPoints = nx * ny
		Y=fft2(vals)/(numPoints)
		#The mean is Y[0,0]
		Y[0,0] = 0
		#plotVals = np.absolute(Y)
		plotVals = np.absolute(fftshift(Y))


def getValsIni:
	import matplotlib.image as img
	magFile = "mag1.png"
	image = img.imread(magFile)
	return image[:,:,2])
