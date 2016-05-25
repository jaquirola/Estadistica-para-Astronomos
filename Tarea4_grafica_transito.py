import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from scipy.stats import chisqprob, chi2
	

#Extraer los datos de un archivo .dat
datos = loadtxt("datos.dat", float)
time = []
flux = []
erro = []
for i in range (0,300):
	time.append(datos[i][0])
	flux.append(datos[i][1])
	erro.append(datos[i][2]) 

plt.errorbar(time, flux, yerr=erro, fmt='o', color = 'k', label = 'Transito exoplaneta')
plt.axvline(x=0.4, lw = float(2.0), ls = 'dashed', color = 'r')
plt.axvline(x=0.7, lw = float(2.0), ls = 'dashed', color = 'r')
plt.text(0.25, 0.9998, r'Intervalo I', fontsize=12.0)
plt.text(0.50, 1.0001, r'Intervalo II', fontsize=12.0)
plt.text(0.71, 0.9998, r'Intervalo III', fontsize=12.0)
plt.xlabel('Tiempo[dias]')
plt.ylabel('Flujo normalizado')
plt.legend(loc = 'lower left')
plt.xlim([0.2,0.8])
plt.show()
