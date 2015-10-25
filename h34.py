import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

c = 3e8
k = 1.3806488e-23
mu = 1.660538921e-27


vmic = 1000.0

def dopplerShift(l,v):
	return l * v / c

#l is wl in A
def printLine(name, l, T, m):
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



printLine('Halpha', 6563, 15000, 1)
printLine('FeI', 6302, 5000, 56)
printLine('FeI', 15648, 6000, 56)

