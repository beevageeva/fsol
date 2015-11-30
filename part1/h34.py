import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

c = 3e8
k = 1.3806488e-23
mu = 1.660538921e-27


vmic = 1000.0

km = 4.67e-3

def dopplerShift(l,v):
	return l * v / c

#para hoja2
#l is wl in A
def printLine2(name, l, T, m):
	print("\nLine: %s , Lambda (A): %.1f, T = %.1f(K), A = %d, m=%.4e kg" % (name, l, T, m, m * mu))
	m*=mu
	l *= 1e-10
	vterm = np.sqrt(k*T/m)
	vtot = np.sqrt(k*T/m + vmic**2)
	print("vterm=%.4e m/s, vtot = %.4e m/s" % (vterm , vtot))
	d1 = dopplerShift(l, vterm) * 1e10
	d2 = dopplerShift(l, vmic) * 1e10
	d3 = dopplerShift(l, vtot) * 1e10
	print("dlTerm = %.4e A, dlMic = %.4e A, dlTot = %.4e A" % (d1,d2,d3))


def magShift(l, g, B):
	return km * l**2 * g * B

#para hoja3
#l is wl in A
def printLine(name, l, T, m, g):
	print("\nLine: %s , Lambda: %.1f A, g = %.1f" % (name, l,g))
	m*=mu
	l *= 1e-10
	vtot = np.sqrt(k*T/m + vmic**2)
	d3 = dopplerShift(l, vtot) * 1e10
	print("T = %.1f K, v = %.4e m/s, dlD = %.4f A" % (T, vtot,d3))
	for B in Barr:
		ms = magShift(l, g, B) * 1e10
		print("B = %.1f G, dlB = %.4f A" % (B, ms))
	print("dlD = dlB <=> B = %.4f G" % (vtot / (c * km * l * g)))
	#print ("PROOF %.4f" % (dopplerShift(l, vtot) / (km * l**2 * g)))
		
#hoja2
#printLine2('Halpha', 6562.8, 15000, 1)
#printLine2('FeI', 6302.5, 5000, 56)
#printLine2('FeI', 15648, 6000, 56)


#hoja3
Barr = [200, 1000, 3000]
printLine('Halpha', 6562.8, 15000, 1,1)
printLine('FeI', 6302.5, 5000, 56, 2.5)
printLine('FeI', 15648, 6000, 56, 3)
