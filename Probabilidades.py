import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from math import *
import scipy.integrate

def binomial(n, x, r):
	a = factorial(n)
        b = factorial(x)
        c = factorial(n-x)
        factor = a/(b*c)
        probabilidad = factor*(r**x)*(1-r)**(n-x)
	return probabilidad

def denominador(n,x):
	f = lambda r: binomial(n, x, r)
	denominador = scipy.integrate.quad(f, 0, 1)
	return denominador

def pdf_cdf(n, x):
	suma = 0
	cdf = []
	pdf = []	
	for r in np.arange(0, 1+0.01, 0.01):
		a = denominador(n, x)
		probabilidad = binomial(n, x, r)/a[0]
		suma = suma + probabilidad*0.01
		cdf.append(suma)
        	z.append(r)
        	pdf.append(probabilidad)
	return (z, cdf, pdf)

def comparacion(n,x):
	a = denominador(n, x)
	f = lambda r: binomial(n, x, r)/a[0]
	prob1 = scipy.integrate.quad(f, 0, 0.5)
	prob2 = scipy.integrate.quad(f, 0.5, 1)
	razon = prob2[0]/prob1[0]
	return razon, prob1[0], prob2[0]	

n = 33
x = 18
cdf = []
pdf = []
z = []
z, cdf, pdf = pdf_cdf(n, x)
a = denominador(n, x)
razon, prob1, prob2 = comparacion(n, x)
print 'El denominador es de la distribucion P(r|X) =', a[0]
print 'P(r>0.5|X)=', prob2, 'P(r<0.5|X)=', prob1, 'P(r>0.5|X)/P(r<0.5|X)=', razon


plt.plot(z, pdf, 'o', label = 'PDF', color = 'r')
plt.xlabel('$r$')
plt.ylabel('$f_{r|X=18}$')
plt.legend(loc = 'upper left')
plt.title('Funcion de densidad de probabilidad')
plt.show()
plt.plot(z, cdf, 'o', label = 'CDF', color = 'r')
plt.xlabel('$r$')
plt.ylabel('$F_{r|X=18}$')
plt.legend(loc = 'upper left')
plt.title('Funcion de distribucion acumulativa')
plt.show()


