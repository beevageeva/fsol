def intTemp(y):
	return (7.0 /9) * (7e13 * y - 187.3616e18)**(9.0/7) * 1.0/7 * 10 **(-13)

print(intTemp(8e7))
print(intTemp(0.27e7))

m = (intTemp(8e7) - intTemp(0.27e7) + 0.27e11) / 8e7

print(m)


print(0.03175 * 0.6087 / (8314 * 2.3417e-12))
