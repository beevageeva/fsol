import numpy as np

MODEL_FILENAME = "m2.txt"
#MODEL_FILENAME = "m1unsorted.txt"


sigma = 5.67e-8
C = 3e+8
G = 6.674e-11
MS = 1.9891e+30
RS = 0.7e+9
A = float(27*24*3600)



def readVals(filename):
	data = np.loadtxt(filename)
	return data[np.argsort(data[:, 1])]

def interpValues(data, v1):
	n = data.shape[0]
	print("n = %d , v1=%e" % (n, v1))
	i = 0
	while i<n and v1>data[i,1]: 
		i+=1
	print("i = %d " % i)
	if v1==data[i,1]:
		print("ROW i ")
		print(data[i,:])
		print("END EXACT")
		res= {'fm': data[i,0], 'pres': data[i,2], 'temp': data[i,3], 'rho': data[i,4]}
		if i<n-1:
			print("ROW i+1 ")
			print(data[i,:])
			print("END GRAD EXACT")
			res['gradT'] = (data[i+1,3] - data[i,3]) / (data[i+1,1] - data[i,1])
		else:
			if i<1:
				print(" !!!!!")
				return None
			print("ROW i - 1 ")
			print(data[i,:])
			print("END GRAD EXACT")
			res['gradT'] = (data[i,3] - data[i-1,3]) / (data[i,1] - data[i-1,1])	
		return res
	else: #i==n or v1>data[i-1,1] and v1<data[i,1]
		if i==n:
			print("v1 not in array BIGGER")
			return None
		else: #between i-1 and i
			print("ROWS i and i-1")
			print(data[i,:])
			print(data[i-1,:])
			print("END BETWEEN")
			res = {}
			res['fm'] = data[i-1,0] + ((v1 - data[i-1,1]) * (data[i,0] - data[i-1,0])) / (data[i,1] - data[i-1,1])
			res['pres'] = data[i-1,2] + ((v1 - data[i-1,1]) * (data[i,2] - data[i-1,2])) / (data[i,1] - data[i-1,1])
			res['temp'] = data[i-1,3] + ((v1 - data[i-1,1]) * (data[i,3] - data[i-1,3])) / (data[i,1] - data[i-1,1])
			res['rho'] = data[i-1,4] + ((v1 - data[i-1,1]) * (data[i,4] - data[i-1,4])) / (data[i,1] - data[i-1,1])
			res['gradT'] = (data[i,3] - data[i-1,3]) / (data[i,1] - data[i-1,1])
			return res

def getForces(data, v1):
	from math import pi
	res = interpValues(data, v1)
	if res is None:
		return
	print("res from table for v1=%e : fm=%e, pres=%e, temp = %e, rho = %e , gradTx = %e" % (v1, res['fm'], res['pres'], res['temp'], res['rho'], res['gradT']))
	Fc = (4.0 * pi**2 * RS ) / A**2 * v1 
	Fg = ((G * MS) / RS**2) * (res['fm'] / v1**2)
	Fprad = (-16.0 * sigma * res['temp'] **3) / (3 * C * res['rho'] * RS) *res['gradT']
	print("gradTr = %e , Fg = %e, Fc = %e, Fprad = %e" % (res['gradT'] / RS, Fg,Fc,Fprad))

def probl1(data):
	getForces(data,0.1)
	getForces(data,0.3)
	getForces(data,1)


def probl2(data):
	#FC = 0.21
	FC = 0.25
	data = data[data[:,1]<FC]
	temp = data[:,3].mean()
	rho = data[:,4].mean()
	X = data[:,5].mean()
	temp9 = temp*10**(-9)
		
	print(temp)
	print(temp9)
	print(rho)
	print(X)

	ergSi = 2.54 *  rho * X**2 * temp9 ** (-2.0/3) * np.exp(-3.37 / temp9**(1.0/3))
	
	n = data.shape[0]
	mc = data[n-1,0] * MS


	print("-----")
	print("r/RS = %e , energy = %e" % (data[n-1,1], ergSi * mc))

data = readVals(MODEL_FILENAME)
#print data[:,0]
probl1(data)
#probl2(data)

