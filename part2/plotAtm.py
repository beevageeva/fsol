import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

R = 8.314 #J / (K * mol)
kB = 1.381e-23
mH = 1.674e-27
#print("kB/mH = %e" % (kB / mH))
#R = kB / mH
gSun = 273.6 #m/s**2
z, temp, mmm = np.loadtxt('atmosphere.dat', unpack=True)


def plotTemp():
	plt.title("Temperature (log scale)")
	#plt.title("Temperature")
	plt.xlabel("z(Mm)")	
	plt.ylabel("T(K)")
	plt.yscale('log')	
	#plt.ticklabel_format(style="sci", axis="y")
	plt.plot(z, temp)
	plt.draw()	
	plt.show()	

def plotMMM():
	plt.title("MMM")
	#plt.title("Temperature")
	plt.xlabel("z(Mm)")	
	plt.plot(z, mmm)
	plt.draw()	
	plt.show()	

def plotMMMLayers():
	#plt.hlines([0.6087, 1.2727], z[z.shape[0]-1], z[0])
	#plt.yticks(list(plt.yticks()[0]) + [0.6087, 1.2727] )
	plt.yticks([0.6087, 1, 1.2727] )
	plt.ylim(0.6, 1.3)
	plt.ylabel('mu (g / mol)')
	plt.grid(True)
	plotWithLayers("Mean molecular weight", mmm)





def plotOnAxisWithLayers(ax, vals, printVals = False, plotExtraTicks=False, plotLegend = True):
	idx = getLayerIndices()	
	ax.plot(z[0:idx[0]], vals[0:idx[0]],'o', markersize=2, mec='none', label='corona')
	ax.plot(z[idx[0]+1:idx[1]], vals[idx[0]+1:idx[1]],'o', markersize=2, mec='none', label='trans')
	ax.plot(z[idx[1]+1:idx[2]], vals[idx[1]+1:idx[2]], 'o', markersize=2, mec='none', label='chrom')
	ax.plot(z[idx[2]+1:idx[3]], vals[idx[2]+1:idx[3]], 'o', markersize=2, mec='none',label='phot')
	ax.plot(z[idx[3]+1:], vals[idx[3]+1:],'o', markersize=2, mec='none', label='int')
	ax.grid(True)
	if printVals:
		print("corona between [%f, %f] Mm values: [%e, %e] K" % (z[0], z[idx[0]], vals[0], vals[idx[0]]) )
		print("transition region between [%f, %f] Mm values: [%e, %e] K" % (z[idx[0]+1], z[idx[1]], vals[idx[0]+1], vals[idx[1]]))
		print("chromosphere between [%f, %f] Mm values: [%e, %e] K" % (z[idx[1]+1], z[idx[2]], vals[idx[1]+1], vals[idx[2]]))
		print("photosphere between [%f, %f] Mm values: [%e, %e] K" % (z[idx[2]+1], z[idx[3]], vals[idx[2]+1], vals[idx[3]]))
		print("solar interior between [%f, %f] Mm values: [%e, %e] K" % (z[idx[3]+1], z[z.shape[0]-1], vals[idx[3]+1], vals[z.shape[0]-1]))
		#print("PV:  temp[idx0=%d] = %e,   temp[idx0+1=%d] %e" %(idx[0], temp[idx[0]], (idx[0]+1), temp[idx[0]+1] ))	
	if plotExtraTicks:
		if (type(ax).__name__ == 'AxesSubplot'):
			print("axes meth" )
			print(dir(ax))
		else:
#			ax.xticks(list(ax.xticks()[0]) + [z[idx[0]], z[idx[1]], z[idx[2]], z[idx[3]]] )
#			ax.yticks(list(ax.yticks()[0]) + [vals[idx[0]], vals[idx[1]], vals[idx[2]], vals[idx[3]]] )
#			ax.grid(True)
			ax.vlines([z[idx[0]], z[idx[1]], z[idx[2]], z[idx[3]], z[0], z[z.shape[0] - 1] ], [0], [vals[idx[0]], vals[idx[1]], vals[idx[2]], vals[idx[3]],  vals[0], vals[z.shape[0] - 1]  ], linestyles='dotted', lw=1 )
			ax.hlines([vals[idx[0]], vals[idx[1]], vals[idx[2]], vals[idx[3]],  vals[0], vals[z.shape[0] - 1] ], [-10], [z[idx[0]], z[idx[1]], z[idx[2]], z[idx[3]], z[0], z[z.shape[0] - 1] ] , linestyles='dotted', lw=1 )
	if plotLegend:	
		ax.legend(bbox_to_anchor=(1.125, 1.125))
		#ax.legend(bbox_to_anchor=(1.125, 0.75))


def plotWithLayers(title, vals, printVals = False):
	plt.title(title)
	plt.xlabel("z(Mm)")
	plotOnAxisWithLayers(plt, vals, printVals)
	plt.draw()	
	plt.show()	

def plotNHDivNeLayers():
	func = mmm / (1.4 - 1.1 * mmm)
	plt.ylim(0,4)
	plt.yticks(list(plt.yticks()[0]) + [func[0]] )
	plotWithLayers("NH/Ne", func, True)
	plt.savefig("nHDivNe.png")

def plotHp():
	plt.ylabel("Hp(Mm)")
	plotWithLayers("Pressure scale height", (R / gSun) * temp / mmm * 1e-6 ) #Plot it in Mm

def getLayerIndicesTemp():
	# desc
	CORONA_START = 500000
	TRANS_REG_START = 8000
	#chromosphere starts in min
	# now asc
	PHOT_START = 6500
	minTemp = np.min(temp)
	res = []	
	n = 0
	i = 0
	val = temp[0]
	while n < 4:
		if n==0:
			#desc order
			if(val < CORONA_START):
				#print("temp[i=%d] = %e,   temp[i-1=%d] %e" %(i, temp[i],(i-1), temp[i-1] ))	
				res.append(i-1)		
				n+=1
		elif n==1:
			#desc order
			if(val < TRANS_REG_START):
				res.append(i-1)		
				n+=1
		elif n ==2:
			if(val == minTemp):
				res.append(i)		
				n+=1
		elif (n==3):
			#asc order
			if(val > PHOT_START):
#			if(z[i]<0):
				res.append(i-1)		
				n+=1
		i+=1
		val = temp[i]
	return res



getLayerIndices = getLayerIndicesTemp
#def getLayerIndicesDz():
#	# desc
#	CORONA_START = 500000
#	TRANS_REG_START = 8000
#	#chromosphere starts in min
#	# now asc
#	PHOT_START = 6500
#	minTemp = np.min(temp)
#	res = []	
#	n = -1
#	i = 1
#	dz = 0
#	while n < 4 and i< z.shape[0]:
#		newdz = z[i-1] - z[i]
#		if abs(newdz - dz)>0.04:
#			if n!=-1:	
#				res.append(i)
#			dz = newdz
#			n+=1
#		i+=1
#	return res
#getLayerIndices = getLayerIndicesDz


def plotTempLayers():
	plt.title("Temperature (log scale)")
	plt.xlabel("z(Mm)")	
	plt.ylabel("T(K)")
	plt.yscale('log')
	print("Temp:")
	plotOnAxisWithLayers(plt, temp, True, True)
	#plotOnAxisWithLayers(plt, temp)
	plt.savefig("tempLayers.png")
	plt.draw()	
	plt.show()	

#zz ordered downwards
def integrateFunc(zz, valF, func):
	i = 0
	p = np.zeros(len(z))
	p[0] = valF
	for i in range(0,len(z)-1):
		dz = zz[i] - zz[i+1]
		p[i+1] = p[i] -  0.5*(func[i+1] + func[i])*dz
	return p

def integrateFuncFw(zz, valI, func):
	#reverse zz ordered downwards
	zz = zz[::-1]
	func = func[::-1]
	i = 0
	p = np.zeros(len(z))
	p[0] = valI
	for i in range(0,len(z)-1):
		dz = zz[i+1] - zz[i]
		p[i+1] = p[i] +  0.5*(func[i+1] + func[i])*dz
	return p[::-1]

def analyticTest():
	#Hp = 1 => p = rho * gSun * Hp => mmm / temp = Hp * R / gSun
	#pf outside, pi inside
	calcFw = False
	#calcFw = True
	zz = z*1e6 #m
	zi = zz[len(z)-1]
	zf = zz[0]
	#Hp = 1e10
	Hp = 1
	for i, rhof in enumerate([1e-10, 1e-5, 1e-2,1, 1e2,1e3, 1e7, 1e10]):
		pfE = rhof * gSun * Hp
		pf = np.log(pfE)
		func = np.zeros(len(z)) - 1 / Hp
		if calcFw:
			pi = pf + 1/Hp * (zf - zi) 
			numPres = integrateFuncFw(zz, pi, func)
		else:
			numPres = integrateFunc(zz, pf, func)
		numRho = numPres -  np.log(gSun * Hp)
		anPres = pf +  (zf-zz) / Hp
		print("log pres at top: %e, log pres at the bottom: %e" % (numPres[0], numPres[z.shape[0]-1])) 
		print("log an pres at top: %e, log an pres at the bottom: %e" % (anPres[0], anPres[z.shape[0]-1])) 
		anRho = np.log(rhof) + (zf-zz) / Hp
		f, (ax1, ax2) = plt.subplots(2, sharex=True)
		ax1.set_title('Hp=%e, rhoF = %e kg/m3, pF = %e Pa' % (Hp, rhof, pfE), y=1.08)
		ax1.set_xlabel('z(Mm)')
		ax1.set_ylabel('ln p(Pa)')
		ax1.plot(z, numPres, 'r-', label="num")
		ax1.plot(z, anPres , 'g-', label="an")
		ax1.legend()
		ax2.set_ylabel('ln rho(kg/m**3)')
		ax2.plot(z, numRho, 'r-', label='num')
		ax2.plot(z, anRho, 'g-', label="an")
		ax2.legend()
		plt.draw()	
		#plt.show()
		if calcFw:
			plt.savefig('analyticFw_%d.png' % i)	
		else:
			plt.savefig('analytic_%d.png' % i)	



def getPresLog(calcFw):
	pF = np.log(5.219e-3) #Pa
	func = -(mmm * 1e-3 * gSun)  / (temp * R)
	if calcFw:
		pI = np.log(4.469126e+06) 
		pres = integrateFuncFw(z*1e6, pI, func)
	else:
		pres = integrateFunc(z*1e6, pF, func)
	return pres

def getRhoLog(pres):
	return pres + np.log(mmm) - np.log(R) - np.log(temp)


def integrateFile():
	calcLog = True
	#calcLog = False
	calcFw = False
	pres = getPresLog(calcFw)
	rho = getRhoLog(pres) 
	B = np.log(1e-3) #measured in T = 10G
	gamma = np.log(5/3)
	from math import pi as mathPi
	mu0 = np.log(4 * mathPi * 10**(-7))
	pmag = 2 * B - np.log(2) - mu0
	betaPl = pres - pmag
	vA = B - 0.5*(mu0 + rho)
	cs = 0.5 * (pres + gamma - rho) 
	difVaCs = vA - cs
	difVaCsBp = betaPl + 2 *  difVaCs + gamma - np.log(2) 
	print("pres at top: %e, pres at the bottom: %e" % (np.exp(pres[0]), np.exp(pres[pres.shape[0]-1]))) 
	print("rho at top: %e, rho at the bottom: %e" % (np.exp(rho[0]), np.exp(rho[pres.shape[0]-1]))) 
	#f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, sharex=True)
	f1, (ax1, ax2) = plt.subplots(2, sharex=True)
	f2, (ax3, ax4, ax5) = plt.subplots(3, sharex=True)
	#f3, (ax6, ax7) = plt.subplots(2, sharex=True)
	f3, (ax7) = plt.subplots(1)
	ax7.set_xlabel('z(Mm)')
	ax5.set_xlabel('z(Mm)')
	ax2.set_xlabel('z(Mm)')
	ax1.set_title('from file')
	if calcLog:
		fileName = "fromFileLn"
	#ln pres and ln rho
		ax1.set_ylabel('ln p(Pa)')
		ax2.set_ylabel('ln rho(kg/m**3)')
		ax3.set_ylabel('ln beta p.')
		ax4.set_ylabel('ln vA (m/s)')
		ax5.set_ylabel('ln cs (m/s)')
		#ax6.set_ylabel('ln vA/cs ')
		ax7.set_ylabel('ln func(bp,vA/cs) ')
		plotOnAxisWithLayers(ax1, pres)
		plotOnAxisWithLayers(ax2, rho, plotLegend=False)
		plotOnAxisWithLayers(ax3, betaPl)
		plotOnAxisWithLayers(ax4, vA, plotLegend=False)
		plotOnAxisWithLayers(ax5, cs, plotLegend = False)
		#plotOnAxisWithLayers(ax6, difVaCs)
		#plotOnAxisWithLayers(ax7, difVaCsBp, plotLegend = False)
		plotOnAxisWithLayers(ax7, difVaCsBp)
	else:
		fileName = "fromFile"
		scaleLog = False
		#scaleLog = True
		#pres and rho
		ax1.set_ylabel('p(Pa)')
		#ax1.set_ylim(0,20000)
		#ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%e'))
		ax2.set_ylabel('rho(kg/m**3)')
		#ax2.set_ylim(0,0.4)
		ax3.set_ylabel('beta plasma')
		ax4.set_ylabel('vA (m/s)')
		ax5.set_ylabel('cs (m/s)')
		#ax6.set_ylabel('cs/vA')
		ax7.set_ylabel('func(bp, cs/vA)')
		if scaleLog:
			fileName = "fromFileLogScale"
			ax1.set_yscale('log')
			ax2.set_yscale('log')
			ax3.set_yscale('log')
			ax4.set_yscale('log')
			ax5.set_yscale('log')
			ax6.set_yscale('log')
			ax7.set_yscale('log')
		plotOnAxisWithLayers(ax1, np.exp(pres), plotLegend=False)
		plotOnAxisWithLayers(ax2, np.exp(rho), plotLegend=False)
		plotOnAxisWithLayers(ax3, np.exp(betaPl), plotLegend=False)
		plotOnAxisWithLayers(ax4, np.exp(vA), plotLegend=False)
		plotOnAxisWithLayers(ax5, np.exp(cs), plotLegend=False)
		#plotOnAxisWithLayers(ax6, np.exp(difVaCs), plotLegend=False)
		plotOnAxisWithLayers(ax7, np.exp(difVaCsBp), plotLegend = False)
	
	f1.savefig("%s1.png" % fileName)
	f2.savefig("%s2.png" % fileName)
	f3.savefig("%s3.png" % fileName)

	#no layers
	#ax1.plot(z, pres)
	#ax2.plot(z, rho)
	plt.draw()	
	plt.show()	


def radiative():
	log10T, lambdaPh, lambdaC = np.loadtxt('dere_etal_table.dat', unpack=True)
	def plotLambdaPh():
		plt.xlabel("log10 T (K)")
		plt.ylabel("LambdaPh (J m**3/s)")
		plt.plot(log10T, lambdaPh)
		plt.grid(True)
		plt.draw()	
		plt.savefig("lambdaPh.png")
		#plt.show()	
	def plotLambdaC():
		plt.xlabel("log10 T (K)")
		plt.ylabel("LambdaC (J m**3/s)")
		plt.plot(log10T, lambdaC)
		plt.grid(True)
		plt.draw()	
		plt.savefig("lambdaC.png")
		#plt.show()	
	def getTempMax():
		i1 = np.argmax(lambdaC)
		i2 = np.argmax(lambdaPh)
		print("i1 = %d, i2 = %d" % (i1, i2))
		print("temp where both  Lambda max = %e" % 10**log10T[i1])

	def interpolateLambdaC():
		atmInd = np.where(temp>3*10**4)
		atmTemp = temp[atmInd]
		atmZ = z[atmInd]
		interpLambdaC = np.interp(np.log10(atmTemp),log10T, lambdaC )
		atmRho = (np.exp(getRhoLog(getPresLog(False))))[atmInd]
		atmNH = atmRho / (1.4 * mH)
		atmNe = 6/5 * atmNH
		atmLr = interpLambdaC *  atmNH * atmNe
		#nt = nh + ne + nhe = nh + nh + 2 nhe + nhe = 2 nh + 3 nhe = 2.3 nh
		atmNt = 2.3 * atmNH
		ue = 1.5 * atmNt * kB * atmTemp			

		def plotInterpValues():
			plt.xlabel("z (Mm)")
			plt.ylabel("LambdaC (J m**3/s)")
			plt.plot(atmZ, interpLambdaC)
			plt.grid(True)
			plt.draw()	
			plt.savefig("interpLambdaC.png")
			plt.show()
		def plotLrValues():
			plt.xlabel("z (Mm)")
			plt.ylabel("Lr (J / m**3*s)")
			plt.plot(atmZ, atmLr)
			plt.grid(True)
			plt.yscale('log')
			plt.draw()	
			plt.savefig("Lr.png")
			plt.show()
		def plotUeValues():
			plt.xlabel("z (Mm)")
			plt.ylabel("ue (J / m**3)")
			plt.plot(atmZ, ue)
			plt.grid(True)
			plt.draw()	
			plt.savefig("ue.png")
			plt.show()
		def plotUeDivLr():
			plt.xlabel("z (Mm)")
			plt.ylabel("ue / Lr")
			plt.plot(atmZ, ue / atmLr)
			plt.grid(True)
			plt.draw()	
			plt.savefig("ueDivLr.png")
			plt.show()
		#plotInterpValues()
		#plotLrValues()
		plotUeValues()
		#plotUeDivLr()
		
	
	
	#plotLambdaPh()
	#plotLambdaC()
	#getTempMax()
	interpolateLambdaC()
	

#plotTemp()
#plotTempLayers()
#plotMMM()
#plotMMMLayers()
#plotNHDivNeLayers()
#plotHp()
#analyticTest()
integrateFile()
#radiative()


