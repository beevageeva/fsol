from scipy import integrate
from math import log,sqrt,sin,cos
import matplotlib.pyplot as plt
import numpy as np

def funcIntegr(r):
    return lambda k: 1.0/(k**5)* ((np.log(1+11.14*k))**2) * ((1+18.5*k + 5880*k**2+17580*k**3 + 1.04e6 * k**4)**(-0.5)) * ((np.sin(k*r) - k*r*np.cos(k*r))**2) 
    #return lambda k: 1.0/(k**5)*(np.log(1+11.14*k))**2 * np.sqrt(1+18.5*k + 5880*k**2+17580*k**3 + 1.04e6 * k**4)*(1.0/3 * (k*r)**3 - 1.0/30 * (k*r)**5 )**2/19845 


def funcIntegr2(r):
   	return lambda k: (np.sin(k*r) - k*r*np.cos(k*r))**2 
    #return lambda k: (1.0/3 * (k*r)**3 - 1.0/30 * (k*r)**5 )**2


N = 100
R = 30.0
#
func = funcIntegr(R)
##func = funcIntegr2(8)
##k = np.arange(1,10000)
k = np.linspace(0,N/R,1024)
##plt.plot(k, func(k), 'o')
plt.plot(k, func(k))
plt.draw()
plt.show()

print("INF")
print(integrate.quad(func, 0, np.inf))
print("quad 1000")
print(integrate.quad(func, 1, 1000))
print("quad 100")
print(integrate.quad(func, 1, 100))
print("quad N/R = %f" % (N/R))
print(integrate.quad(func, 1, N/R))
print("romberg 1000")
print(integrate.romberg(func, 1, 1000))
print("romberg 100")
print(integrate.romberg(func, 1, 100))

integral, error = integrate.quad(func, 1, 1000)

from math import sqrt,pi

print("----------------sigma8")
print(sqrt(( (0.83)**2 * R**6 * 2 * pi**2) / (9*19843*integral)))
print("----------------coef")
print(  (  R**6 * 2 * pi**2) / (9*integral))

print("----------------sigma8")
print(sqrt(( (0.83)**2 * R**6 * 2 * pi**2) / (9*9229390.05*integral)))


