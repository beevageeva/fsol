from scipy import integrate
from math import log,sqrt,sin,cos
import numpy as np

def funcIntegr(r):
    return lambda k: 1.0/(k**5)*(log(1+11.14*k))**2 * sqrt(1+18.5*k + 5880*k**2+17580*k**3 + 1.04e6 * k**4)*(sin(k*r) - k*r*cos(k*r))**2/19845 

print(integrate.quad(funcIntegr(8), 0, np.inf))
