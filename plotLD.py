import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

h = 6.62607004e10-34
c = 3e8
k = 1.3806488e-23

#l is lambda array in nanometers: see ml mult
def planckFunc(l, T):
	ml = 1e-9
	l*=ml
	return (2.0 *  h * c**2)/(l**5 * (np.exp((h*c)/(k * l * T))-1))

def plotLD(T, startLambda, endLambda):
	numPoints = 100
	x = np.linspace(startLambda, endLambda, numPoints, endpoint=True)
	#pure = planckFunc(T, x) 
	pure = -5.0 / 160000 * (x- startLambda + 100)*(x - endLambda)
	noise = np.concatenate((np.random.normal(0, 1, numPoints/2), np.random.normal(0, 0.25, numPoints/2)))
	print("WITHOUT NOISE")
	print(pure)
	print("NOISE")
	print(noise)
	y = pure + noise	
	plt.plot(x, y, "r-")
	plt.plot(x, 4.0 / 5.0 * y, "g-")
	
	plt.draw()
	plt.show()


plotLD(5000, 100, 900)

