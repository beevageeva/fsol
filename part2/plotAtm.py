import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

R = 8.314 #J / (K * mol)
#R = 8314 #J / (K * mol)
#kB = 1.381e-23
#mH = 1.674e-27
#print("kB/mH = %e" % (kB / mH))
#R = kB / mH
gSun = 2.736e-4 #m/s**2
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
	plt.grid(True)
	plotWithLayers("MMM", mmm)





def plotOnAxisWithLayers(ax, vals, printVals = False, plotExtraTicks=False):
	idx = getLayerIndices()	
	ax.plot(z[0:idx[0]], vals[0:idx[0]],'o', markersize=2, mec='none', label='corona')
	ax.plot(z[idx[0]+1:idx[1]], vals[idx[0]+1:idx[1]],'o', markersize=2, mec='none', label='trans')
	ax.plot(z[idx[1]+1:idx[2]], vals[idx[1]+1:idx[2]], 'o', markersize=2, mec='none', label='chrom')
	ax.plot(z[idx[2]+1:idx[3]], vals[idx[2]+1:idx[3]], 'o', markersize=2, mec='none',label='phot')
	ax.plot(z[idx[3]+1:], vals[idx[3]+1:],'o', markersize=2, mec='none', label='int')
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
	ax.legend(bbox_to_anchor=(1.125, 1.125))


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
	plt.grid(True)
	plotWithLayers("NH/Ne", func, True)

def plotHp():
	plotWithLayers("Hp", (R / gSun) * temp / mmm )

def getLayerIndices():
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

def plotTempLayers():
	plt.title("Temperature (log scale)")
	plt.xlabel("z(Mm)")	
	plt.ylabel("T(K)")
	plt.yscale('log')
	print("Temp:")
	plotOnAxisWithLayers(plt, temp, True, True)
	#plotOnAxisWithLayers(plt, temp)
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

#zz ordered downwards
def integrateFuncFw(zz, valI, func):
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
	zz = z*1e6 #m
	zi = zz[len(z)-1]
	zf = zz[0]
	Hp = 1
	for i, rhof in enumerate([1e-10, 1e-4, 1e-2,1, 1e2,1e5, 1e8, 1e9, 1e10]):
		pf = rhof * gSun * Hp
		pi = np.log(pf) + 1/Hp * (zf - zi) 
		func = np.zeros(len(z)) - 1 / Hp
		#numPres = integrateFunc(zz, pf, func)
		numPres = integrateFuncFw(zz, pi, func)
		numRho = numPres -  np.log(gSun * Hp)
		anPres = np.log(pf) +  (zf-zz) / Hp
		anRho = np.log(rhof) + (zf-zz) / Hp
		f, (ax1, ax2) = plt.subplots(2, sharex=True)
		ax1.set_title('rhoF = %e kg/m3, pF = %e Pa' % (rhof, pf))
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
		plt.savefig('analyticFw_%d.png' % i)	



def integrateFile():
	pF = np.log(5.219e-3) #Pa
	func = -(mmm * gSun)  / (temp * R)
	pres = integrateFunc(z*1e6, pF, func)
	pI = np.log(5.327457e-03) 
	#pres = integrateFuncFw(z*1e6, pI, func)
	print("pres at top: %e, pres at the bottom: %e" % (np.exp(pres[0]), np.exp(pres[pres.shape[0]-1]))) 
	rho = pres + np.log(mmm) - np.log(R) - np.log(temp)
	f, (ax1, ax2) = plt.subplots(2, sharex=True)
	ax1.set_xlabel('z(Mm)')
	ax1.set_title('from file')
	#ln pres and ln rho
	ax1.set_ylabel('ln p(Pa)')
	ax2.set_ylabel('ln rho(kg/m**3)')
	plotOnAxisWithLayers(ax1, pres)
	plotOnAxisWithLayers(ax2, rho)
	#pres and rho
#	ax1.set_ylabel('p(Pa)')
#	ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%e'))
#	ax2.set_ylabel('rho(kg/m**3)')
#	plotOnAxisWithLayers(ax1, np.exp(pres))
#	plotOnAxisWithLayers(ax2, np.exp(rho))

	#no layers
	#ax1.plot(z, pres)
	#ax2.plot(z, rho)
	plt.draw()	
	plt.show()	


#plotTemp()
#plotTempLayers()
#plotMMM()
#plotMMMLayers()
#plotNHDivNeLayers()
#plotHp()
#analyticTest()
integrateFile()


