from scipy import integrate
from math import log,sqrt,sin,cos
import matplotlib.pyplot as plt
import numpy as np

def funcIntegr(r):
    return lambda k: 1.0/(k**5)*(np.log(1+11.14*k))**2 * np.sqrt(1+18.5*k + 5880*k**2+17580*k**3 + 1.04e6 * k**4)*(np.sin(k*r) - k*r*np.cos(k*r))**2 
    #return lambda k: 1.0/(k**5)*(np.log(1+11.14*k))**2 * np.sqrt(1+18.5*k + 5880*k**2+17580*k**3 + 1.04e6 * k**4)*(1.0/3 * (k*r)**3 - 1.0/30 * (k*r)**5 )**2/19845 


def funcIntegr2(r):
   	return lambda k: (np.sin(k*r) - k*r*np.cos(k*r))**2 
    #return lambda k: (1.0/3 * (k*r)**3 - 1.0/30 * (k*r)**5 )**2


#N = 500
#
#func = funcIntegr(10)
##func = funcIntegr2(8)
##k = np.arange(1,10000)
#k = np.arange(0.001,N)
##plt.plot(k, func(k), 'o')
#plt.plot(k, func(k))
#plt.draw()
#plt.show()

#print(func(0))
#print(func(1))
#
#for i in range(1,N):
#	if(func(i)<0):
#		print("i=%d"%i)
print("INF")
print(integrate.quad(funcIntegr(8), 0, np.inf))
print("quad 1000")
print(integrate.quad(funcIntegr(8), 1, 1000))
print("quad 100")
print(integrate.quad(funcIntegr(8), 1, 100))
print("romberg 1000")
print(integrate.romberg(funcIntegr(8), 1, 1000))
print("romberg 100")
print(integrate.romberg(funcIntegr(8), 1, 100))
