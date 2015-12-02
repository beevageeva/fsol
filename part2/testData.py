import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


R = 8.314 #J / (K * mol)
kB = 1.381e-23
mH = 1.674e-27
#print("kB/mH = %e" % (kB / mH))
#R = kB / mH
gSun = 273.6 #m/s**2
z, temp, mmm = np.loadtxt('atmosphere.dat', unpack=True)



#zz ordered downwards
def integrateFunc(zz, valF, func):
	i = 0
	p = np.zeros(len(zz))
	p[0] = valF
	for i in range(0,len(zz)-1):
		dz = zz[i] - zz[i+1]
		p[i+1] = p[i] -  0.5*(func[i+1] + func[i])*dz
	return p

def integrateFuncFw(zz, valI, func):
	#reverse zz ordered downwards
	zz = zz[::-1]
	func = func[::-1]
	i = 0
	p = np.zeros(len(zz))
	p[0] = valI
	for i in range(0,len(zz)-1):
		dz = zz[i+1] - zz[i]
		p[i+1] = p[i] +  0.5*(func[i+1] + func[i])*dz
	return p[::-1]


def getPresLog(calcFw, zz, mmm, temp):
	pF = np.log(5.219e-3) #Pa
	func = -(mmm * 1e-3 * gSun)  / (temp * R)
	if calcFw:
		pI = np.log(4.469126e+06) 
		pres = integrateFuncFw(zz*1e6, pI, func)
	else:
		pres = integrateFunc(zz*1e6, pF, func)
	return pres


def getNew2(z, temp, mmm):
	return (z[::2], temp[::2], mmm[::2]	)

def getNew1(z, temp,mmm):
	newz = np.concatenate((z, (z[1:] + z[:-1]) / 2)) # we still need to do this unfortunatly.
	newz = -np.sort(-newz)
	#newz.sort(kind='mergesort')
	#newz = newz[::-1]
	#use lineal interpolation
	#funcT = interp1d(z, temp)
	#funcMMM = interp1d(z, mmm)
	#use cubic interpolation
	funcT = interp1d(z, temp, kind='cubic')
	funcMMM = interp1d(z, mmm, kind='cubic')
	return (newz, funcT(newz), funcMMM(newz))

#calcFw = True
calcFw = False

insertPoints = True

plt.xlabel("z (Mm)")
plt.ylabel("log p1 ")
pres = getPresLog(calcFw, z, mmm, temp)


if insertPoints:
	z2, temp2, mmm2 = getNew1(z, temp, mmm)
	z2, temp2, mmm2 = getNew1(z2, temp2, mmm2)
else:
	z2, temp2, mmm2 = getNew2(z, temp, mmm)
	z2, temp2, mmm2 = getNew2(z2, temp2, mmm2)
	z2, temp2, mmm2 = getNew2(z2, temp2, mmm2)
	z2, temp2, mmm2 = getNew2(z2, temp2, mmm2)

pres2 = getPresLog(calcFw, z2, mmm2, temp2)

plt.clf()

print("len z %d" % len(z))
print("len2 z %d" % len(z2))

if insertPoints:
	plt.plot(z2, pres2,'ro', markersize=2, mec='none', label='wp')
	plt.plot(z, pres,'o', markersize=2, mec='none', mew='none', label="O")
	#plt.plot(z, pres,'go', markersize=1, mec='none',  label='O')
else:
	plt.plot(z, pres,'o', markersize=2, mec='none', mew='none', label="O")
	#plt.plot(z, pres,'go', markersize=1, mec='none',  label='O')
	plt.plot(z2, pres2,'ro', markersize=2, mec='none', label='wp')


plt.grid(True)
plt.savefig('testData.png')


plt.draw()	
plt.show()	
plt.legend()

