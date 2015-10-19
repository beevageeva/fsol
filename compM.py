import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

MODEL_FILENAME1 = "m1.txt"
#MODEL_FILENAME1 = "m2.txt"
#MODEL_FILENAME1 = "m1unsorted.txt"
MODEL_FILENAME2 = "table1.txt"


sigma = 5.67e-8
C = 3e+8
G = 6.674e-11
MS = 1.9891e+30
RS = 0.7e+9
A = float(27*24*3600)



def comp(field):
	data1 = np.loadtxt(MODEL_FILENAME1)
	data2 = np.loadtxt(MODEL_FILENAME2)
	data1= data1[np.argsort(data1[:, 1])]
	data2= data2[np.argsort(data2[:, 1])]

	#print(data1)
	#print("----------------------")
	#print(data2[data2[:,1]==data1[:,1]])
	plt.plot(data1[:,1], data1[:,field], "ro")
	plt.plot(data2[:,1], data2[:,field], "go", markersize=1)
	plt.draw()
	plt.show()


comp(0)
comp(2)
comp(3)
comp(4)

