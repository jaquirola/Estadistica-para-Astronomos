import numpy as np
import matplotlib.pyplot as plt
from math import *
import random
from scipy.linalg import *
from scipy.special import erfinv
import numpy.random as nr

#Se realizo dos tipos de distribuciones, una considerando la demostracion de la
# tarea, y otra usando una funcion de numpy.

#Creamos vector entre -5 y 5
def vector(n):
	x = []
	x = np.linspace(-5, 5, n+1)
	return x
#Genera una matriz de covarianza (cov) corregido con epsilon
def covarianza(x,n):
	i=j=0
	eps = 1e-10
	sigma = np.zeros([n,n])
	I = np.identity(n)
	for i in range(0,n,1):
		for j in range(0,n,1):
			z = exp(-0.5*((abs(x[i]-x[j]))**2.0))
			sigma[i,j] = z
	cov = sigma + I*eps
	return cov

#Crea numeros aleatorios entre 0 y 1
def aleatorio_uniforme():
	for i in range(0, n):
		z = random.random()
	return z
#Distribuye los numeros aleatorios como normal estándar
def aleatorio_normal(n):
	f = []
	g = []
	suma = 0
	for i in range(0,n):
		z = aleatorio_uniforme()	
		g.append(z)
		w = sqrt(2)*erfinv(2*z-1)
		f.append(w)
	return f
#Genera los X, considerando la matriz de covarianza y el vector S
def normal_multi(n):
	x = vector(n)
	cov = covarianza(x,n)
	L = np.linalg.cholesky(cov)
	S = aleatorio_normal(n)
	X = np.dot(L,S)
	return X

n = 100
#Las siguiente dos variables se las utiliza para evaluar la funcion de numpy
#(numpy).
mu=np.zeros([n])
sigma = covarianza(vector(n),n)
numpy = nr.multivariate_normal(mu, sigma, size=101)
#Evalua la normal multivariada por la dimension de n propuesta en la tarea
f = []
for i in range(0,100):
	X = normal_multi(n)
	f.append(X)
	
#Histograma considerando la demostracion de la tarea
plt.hist(f, alpha=0.5, bins=15, color = 'w')
plt.xlabel('Bins numeros aleatorios')
plt.ylabel('Frecuencia')
plt.title('Histograma: $X$ $\sim N(0,\Sigma)$')
plt.savefig('histograma_multivariada.eps')
#plt.show()

#Histograma considerando la funcion numpy
plt.hist(numpy, alpha=0.1, bins=15)
plt.xlabel('Bins numeros aleatorios')
plt.ylabel('Frecuencia')
plt.title('Histograma: $X$ $\sim N(0,\Sigma)$ generado con la funcion numpy')
plt.savefig('histograma_multivariada_numpy.eps')
#pyplot.show()
