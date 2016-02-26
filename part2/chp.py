import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

mH = 1.674e-24 #g
log10T, lambdaC, lambdaPh = np.loadtxt('dere_etal_table.dat', unpack=True)
def plotLambda():
	plt.plot(log10T, lambdaC)
	plt.plot(log10T, lambdaPh)
	plt.draw()
	plt.show()

def printLambda6():
	lambdaCT0 = np.interp(6,log10T, lambdaC )
	lambdaPhT0 = np.interp(6,log10T, lambdaPh )
	print("Lambda C (10**6 K) = %e , Lamda Ph(10**6 K) = %e " % (lambdaCT0, lambdaPhT0))


def plotFr(T1, T2, title):
	H = 5e9 #50 Mm in cm
	numPoints = 10000
	nhMin = 1e8
	nhMax = 5e9
	nh = np.linspace(nhMin, nhMax, numPoints)
	if title=="phot":
		lambdaArray = lambdaPh
	else:
		lambdaArray = lambdaC
	lambdaT1 = np.interp(np.log10(T1), log10T, lambdaArray)
	lambdaT2 = np.interp(np.log10(T2), log10T, lambdaArray)
	arg = 0.5/0.83 * H * (nhMax**2 - nh**2)
	f1 = lambdaT1 * arg
	f2 = lambdaT2 * arg

	plt.xlabel("nH (cm^-3)")
	plt.ylabel("Fr (erg cm^-2 s^-1) ")
	#f, (ax1, ax2) = plt.subplots(2, sharex=True)
	plt.title(title)
	l1 = plt.plot(nh, f1 , label="T=%e" % T1)
	l1[0].axes.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0e'))
	#plt.ticklabel_format(style='sci')
	plt.plot(nh, f2 , label="T=%e" % T2)
	testValues = [1e5,1e4,5e6]
	density	 = np.loadtxt("density.txt")
	rho = density[1] * 1e-3 #transform to g/cm3
	nHModel = rho / (1.4 * mH)
	print(title)

	nhInterpVals1 = np.interp(testValues, f1[::-1], nh[::-1]) 
	nhInterpVals2 = np.interp(testValues, f2[::-1], nh[::-1]) 

	#print(f1.max())
	#print(f2.max())
	#print(nhInterpVals1)
	#print(nhInterpVals2)
	
#	ind1 = []
#	ind2 = []
#	for val in testValues:
#		ind1.append((np.abs(f1-val)).argmin())
#		ind2.append((np.abs(f2-val)).argmin())

	print("T=%e" % T1)

#	for ii in ind1:
#		n1 = nh[ii]
#		i1 = np.abs(nHModel - n1).argmin()
#		print("Lr(nH = %e) = %e ; in model: nH = %e at z = %e Mm" % (n1, f1[ii], nHModel[i1], density[0][i1]))

	for ii in range(len(nhInterpVals1)):
		n1 = nhInterpVals1[ii]
		i1 = np.abs(nHModel - n1).argmin()
		print("Lr(nH = %e) = %e ; in model: nH = %e at z = %e Mm" % (n1, testValues[ii], nHModel[i1], density[0][i1]))

	print("T=%e" % T2)

#	for ii in ind2:
#		n1 = nh[ii]
#		i1 = np.abs(nHModel - n1).argmin()
#		print("Lr(nH = %e) = %e ; in model: nH = %e at z = %e Mm" % (n1, f2[ii], nHModel[i1], density[0][i1]))

	for ii in range(len(nhInterpVals2)):
		n1 = nhInterpVals2[ii]
		i1 = np.abs(nHModel - n1).argmin()
		print("Lr(nH = %e) = %e ; in model: nH = %e at z = %e Mm" % (n1, testValues[ii], nHModel[i1], density[0][i1]))

	plt.grid(True)
	plt.grid(True)
	fig1 = plt.gcf()
	plt.legend()
	plt.draw()
	plt.show()
	fig1.savefig("Fr%s" % title)
	

plotFr(1e6, 2e6, "corona")
plotFr(1e6, 2e6, "phot")


