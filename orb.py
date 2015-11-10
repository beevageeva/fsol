from fractions import Fraction


def calcG(S,L,J):
	if J == 0:
		return 0
	return 1 + Fraction( J*(J+1) + S*(S+1) - L*(L+1) , 2*J*(J+1))


def calcGEf(g1,g2,J1,J2):
	return Fraction(1,2)*(g1+g2)+Fraction(1,4)*(g1-g2)* (J1*(J1+1)-J2*(J2+1) )


def printAll(S1,L1,J1, S2,L2,J2):
	print("(%d,%d,%d) - (%d,%d,%d)" % (S1,L1,J1,S2,L2,J2))
	g1 =  calcG(S1,L1,J1)
	g2 =  calcG(S2,L2,J2)
	print("g1=" + str(g1))
	print("g2=" + str(g2))
	print("gef = " + str(calcGEf(g1,g2,J1,J2)))

printAll(2,2,2,3,2,3)
printAll(2,2,0,3,2,1)
printAll(2,3,1,2,2,0)
printAll(2,1,2,2,2,2)
printAll(2,1,1,2,2,0)
