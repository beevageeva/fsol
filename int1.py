from __future__ import division
from sympy import symbols, Symbol, S, integrate, oo, exp
x = Symbol('x')
#x, y, z, t = symbols('x y z t')
#k, m, n = symbols('k m n', integer=True)
#f, g, h = symbols('f g h', cls=Function)

#print(integrate(x**3/(exp(x)-1), (x,0,oo)))
print(integrate(x**S(3)/(exp(x)-1), x))

