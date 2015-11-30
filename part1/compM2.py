import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

MODEL_FILENAME1 = "table1.txt"
MODEL_FILENAME2 = "table2.txt"



data1 = np.loadtxt(MODEL_FILENAME1)
data2 = np.loadtxt(MODEL_FILENAME2)
#only common
data1 = data1[data1[:,1]>=np.min(data2[:,0])]
data1= data1[np.argsort(data1[:, 1])]
print("--------data1--------")
print(data1)
print("--------data1 end--------")
data2= data2[np.argsort(data2[:, 0])]


def comp(field1, field2, title):
	plt.title(title)
	#print(data1)
	#print("----------------------")
	#print(data2[data2[:,1]==data1[:,1]])
	plt.plot(data1[:,1], data1[:,field1], "r-", label="no conv")
	plt.plot(data2[:,0], data2[:,field2], "g-", label="with conv")
	plt.legend()
	plt.draw()
	plt.show()


comp(2,1, "pres")
comp(3,2, "temp")
comp(4,3, "rho")

