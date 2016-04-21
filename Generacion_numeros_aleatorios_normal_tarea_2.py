import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from math import *
import random
from scipy.special import erfinv

def aleatorio_uniforme():
	for i in range(0, n):
		z = random.random()
	return z

def aleatorio_normal(n):
	f = []
	g = []
	suma = 0
	for i in range(0,n):
		z = aleatorio_uniforme()	
		g.append(z)
		w = sqrt(2)*erfinv(2*z-1)
		f.append(w)
	return g, f

n = 10000
z, f = aleatorio_normal(n)
#Genera histograma de numeros aleatorios uniformes.
plt.hist(z, bins=20, normed=True, color = 'w')
plt.xlabel('Bins numeros aleatorios')
plt.ylabel('Frecuencia')
plt.title('Histograma: 10000 numeros aleatorios $\sim U(0,1)$')
plt.savefig('histograma_uniforme.eps')
plt.clf()	
plt.close
#Crea histograma de numeros aleatorios normal estandar distribucion
plt.hist(f, bins=20, normed=True, color = 'w')
x = np.linspace(-5.0, 5.0)
plt.plot(x, st.norm.pdf(x, 0, np.sqrt(1)), label = '$\sim$Normal(0,1)', color = 'r')
plt.xlabel('Bins numeros aleatorios')
plt.ylabel('Frecuencia')
plt.title('Histograma: 10000 numeros aleatorios $\sim N(0,1)$')
plt.legend(loc = 2)
plt.savefig('histograma_normal.eps')
