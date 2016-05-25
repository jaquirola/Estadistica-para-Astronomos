import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from scipy.stats import chisqprob, chi2

#Funcion para calcular la matriz de covarianza
def covarianza(sigma, r):
	C = np.zeros([r,r])
	for i in range (0,r,1):
		for j in range (0,r,1):
			if (i == j):
				C[i,j] = sigma**2
	return C
#Funcion para obtener los parametros de ajuste theta y las incertidumbres
def ajuste (M,C,Y,p):
	theta = np.zeros([p+2,1])
	MT = M.transpose()
	CI = inv(C)
	prev = np.dot(MT,CI)
	incertidumbre = inv(np.dot(prev,M))
	prev2 =np.dot(prev,Y)
	theta = np.dot(incertidumbre, prev2)
	return incertidumbre, theta
#FUncion para encontrar chi-cuadrado
def chi_cuadrado(Y,M,theta,C):
	x1 = Y-np.dot(M,theta)
	z = np.dot(x1.transpose(),inv(C))
	chi = np.dot(z,x1)
	return chi
#FUncion para obtener los p-values
def p_values(chi,r, p):
	pvalues = chisqprob(chi,r-p)
	return pvalues
#FUncion de la matriz de diseno
def matriz_diseno(dimension,p):
	M = np.zeros([dimension,p+2])
	for i in range (0,dimension,1):
		for j in range (0,p+2,1):
			if (time[i] < 0.4):
				if(j==0):
					M[i,j] = 0
				else:
					M[i,j] = (time[i])**(j-1) 
			elif (0.4 <= time[i] <= 0.7):
				if (j==0):
					M[i,j] = 1
				else:
					M[i,j] = (time[i])**(j-1)  
			else:
				if (j==0):
					M[i,j] = 0
				else:
					M[i,j] = (time[i])**(j-1) 
	return M

#Extraer los datos de un archivo .dat
datos = loadtxt("datos.dat", float)
time = []
flux = []
erro = []
for i in range (0,len(datos)):
	time.append(datos[i][0])
	flux.append(datos[i][1])
	erro.append(datos[i][2]) 

#Rango del polinomio
p = 10
#Sigma de los datos 
sigma = 30e-6
flujo = []
for i in range(0,len(flux)):
	x = flux[i]-1
	flujo.append(x)


#RAngo
polinomio = []
likelihood = []
AIC = []
BIC = []
for k in range (0,p+1):
#Matriz de incertidumbres
	C = covarianza(sigma,len(flujo))
#Matriz de diseno
	M=matriz_diseno(len(flujo),k)
	incertidumbre, theta = ajuste(M, C, flujo, k)
	resultado = []
	L = 0
	for i in range (0,len(time)):
		u = 0
		for t in range(0,len(theta)):
			if (0.4 < time[i] < 0.7):
				if (t==0):
					u = u + theta[t]
				else:
					u = u + theta[t]*(time[i]**(t-1))
			else:
				if (t==0):
					u = 0
				else:
					u = u + theta[t]*(time[i]**(t-1))
		resultado.append(1+u)
		a = ((flujo[i]-u)**2)
		b = a/float(2*sigma**2)
		c = log(1/sqrt(2*pi*sigma**2))
		L = L + (c - b)
	likelihood.append(L)
	polinomio.append(k)
	x = -2*L + 2*(k+2) + (2*(k+2)*(k+3)/(300-k+1))
	y = -2*L + (k+2)*log(300)
	AIC.append(x)
	BIC.append(y)

#Plot al fittear todo el flujo
plt.plot(time, resultado, '-', color = 'b', lw = 3, label = 'Funcion de ajuste, Transito + p='+str(p))
plt.errorbar(time, flux, yerr=erro, fmt='o', color = 'k', label = 'Transito exoplaneta')
plt.xlabel('$\mathrm{Tiempo[dias]}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc = 'lower left')
plt.xlim([0.15,0.85])
plt.show()
plt.close()
#AIC y BIC
plt.plot(polinomio, AIC, '-', color = 'b', lw = 3, label = 'AIC')
plt.plot(polinomio, BIC, '-', color = 'r', lw = 3, label = 'BIC')
plt.xlabel('$\mathrm{p}$')
plt.ylabel('$\mathrm{Information}$ $\mathrm{criteria}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()
