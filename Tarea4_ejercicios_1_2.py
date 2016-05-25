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
#Funcion para obtener los parametros de ajuste theta 
#y las incertidumbres
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
#Extraer los datos de un archivo .dat y guardar en 
#diferentes arrays: flux, time, error
datos = loadtxt("datos.dat", float)
time = []
flux = []
erro = []
for i in range (0,len(datos)):
	time.append(datos[i][0])
	flux.append(datos[i][1])
	erro.append(datos[i][2]) 

#Rango del polinomio
p = 5
#Sigma de los datos 
sigma = 30e-6
flujo = []
for i in range(0,len(flux)):
	x = flux[i]-1
	flujo.append(x)
#Matriz de incertidumbres
C = covarianza(sigma,len(flujo))
#Matriz de diseno
M=matriz_diseno(len(flujo),p)
#Matriz de incertidumbres
incertidumbre, theta = ajuste(M, C, flujo, p)
#Se evalua los parametros obtenidos a diferentes valores de x
resultado = []
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
print 'Ejercicio 1'
print 'Parte b y c'
#Ejercicio 1
#Valores de parametros y varianza
print 'Estimadores de los parametros y su desviacion estandar'
for i in range (0,len(theta)):
	if(i==0):
		print 'delta' ,theta[i], '+/-', sqrt(incertidumbre[i][i])
	else:
		print 'theta'+str(i-1) ,theta[i], '+/-', sqrt(incertidumbre[i][i])
#CHi-cuadrado y p-values
print 'Valores de x^2 y p-values para el transito'
Chi_cuadrado = chi_cuadrado(flujo,M,theta,C)
pvalues = p_values(Chi_cuadrado,len(flujo),p+2)
print '$X^2$=', Chi_cuadrado
print 'P_values=', pvalues

#Plot al fittear todo el flujo
plt.plot(time, resultado, '-', color = 'b', lw = 3, label = 'Funcion de ajuste, Transito + p='+str(p))
plt.errorbar(time, flux, yerr=erro, fmt='o', color = 'k', label = 'Transito exoplaneta')
plt.xlabel('$\mathrm{Tiempo[dias]}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc = 'lower left')
plt.xlim([0.15,0.85])
plt.show()
plt.close()

print ' '
print 'Ejercicio2'
print 'Parte a'

####################################################################################
#Ejercicio2
#Vector de p-values
sigma = 30e-6
pvalues2 = []
#RAngo del polinomio
p = 5
chi2 = []
#Simulacion de 1000 set de datos 
for j in range (0,1000):
	chi = 0
	simulado = []
	curva = []
	for i in range (0,len(time)):
		y = 0
		u = 0
		#FUncion de ajuste
		u = u + theta[1]*(time[i]**(0)) + theta[2]*(time[i]**(1)) + theta[3]*(time[i]**(2)) + theta[4]*(time[i]**(3)) + theta[5]*(time[i]**(4)) + theta[6]*(time[i]**(5))
		#Datos simulados
		y = u + float(np.random.normal(0,30e-6,1))
		simulado.append(y+1)
		curva.append(u+1) 
#	pvalues.append(p_values(chi,len(simulado),p+1))
#	chi2.append(chi)
#Matriz de covarianza
	C = covarianza(sigma,len(simulado))
#Matriz de diseno para polinomio 5
	M = np.zeros([len(simulado),p+1])
	for i in range (0,len(simulado),1):
		for t in range (0,p+1,1):
			M[i,t] = time[i]**t 
	incertidumbre, parametros = ajuste(M, C, simulado, p-1)	
	Chi_cuadrado = chi_cuadrado(simulado,M,parametros,C)
	q = p_values(Chi_cuadrado,len(simulado),p+1)
	pvalues2.append(q)

resultado = []
#Evaluar el ultimo ajuste
for i in range(0,len(time)):
	u = 0
	for t in range(0,len(parametros)):
		u = u + parametros[t]*time[i]**t
	resultado.append(u)
	
#Histograma de p-values ajustando
plt.hist(pvalues2, bins=20, normed=True, color = 'w', alpha=0.5)
plt.title('Histograma de p-values de 10000 simulaciones (p=5)')
plt.xlabel('$\mathrm{Bin}$ $\mathrm{p-values}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'upper left')
plt.show()
plt.close()
#Plot de la simulacion y los ajustes de la ultima simulacion
plt.errorbar(time, simulado, yerr=erro, fmt='o',color = 'b', label = 'Simulacion (funcion p=5)')
plt.plot(time,resultado, color = 'r', label = 'Funcion de ajuste (p=5)')
plt.xlabel('$\mathrm{Tiempo[dias]}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

####################################################################################
print 'Parte c'
#Ejercicio2
#Vector de p-values
sigma = 30e-6
pvalues = []
pvalues2 = []
#RAngo del polinomio
p = 8
chi2 = []
#Simulacion de 1000 set de datos 
for j in range (0,1000):
	chi = 0
	simulado = []
	curva = []
	for i in range (0,len(time)):
		y = 0
		u = 0
		#FUncion de ajuste
		u = u + theta[1]*(time[i]**(0)) + theta[2]*(time[i]**(1)) + theta[3]*(time[i]**(2)) + theta[4]*(time[i]**(3)) + theta[5]*(time[i]**(4)) + theta[6]*(time[i]**(5))
		#Datos simulados
		y = u + float(np.random.normal(0,30e-6,1))
		simulado.append(y+1)
		curva.append(u+1) 
#Matriz de covarianza
	C = covarianza(sigma,len(simulado))
#Matriz de diseno para polinomio 3
	M = np.zeros([len(simulado),p+1])
	for i in range (0,len(simulado),1):
		for t in range (0,p+1,1):
			M[i,t] = time[i]**t 
	incertidumbre, parametros = ajuste(M, C, simulado, p-1)	
	Chi_cuadrado = chi_cuadrado(simulado,M,parametros,C)
	q = p_values(Chi_cuadrado,len(simulado),p+1)
	pvalues2.append(q)
print parametros
print incertidumbre
resultado = []
#Evaluar el ultimo ajuste
for i in range(0,len(time)):
	u = 0
	for t in range(0,len(parametros)):
		u = u + parametros[t]*time[i]**t
	resultado.append(u)
	
#Histograma de p-values ajustando
plt.hist(pvalues2, bins=20, normed=True, color = 'w', alpha=0.5)
plt.title('Histograma de p-values de 1000 simulaciones (p=8)')
plt.xlabel('$\mathrm{Bin}$ $\mathrm{p-values}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'upper left')
plt.xlim(0, 0.5)
plt.show()
plt.close()
#plot datos simulados y ajuste
plt.errorbar(time, simulado, yerr=erro, fmt='o',color = 'b', label = 'Simulacion (funcion p=5)')
plt.plot(time,resultado, color = 'r', label = 'Funcion de ajuste (p=1)')
plt.xlabel('$\mathrm{Tiempo[dias]}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()


#############################################################################################
print 'Ejercicio 2'
print 'parte d'
#Vector de p-values
sigma = 30e-6
pvalues2 = []
#RAngo del polinomio
p = 5
chi2 = []
#Simulacion de 1000 set de datos 
for j in range (0,1000):
	chi = 0
	simulado = []
	curva = []
	for i in range (0,len(time)):
		y = 0
		u = 0
		for t in range (0,len(theta)):
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
		#Datos simulados
		y = u + float(np.random.normal(0,sigma,1))
		simulado.append(y+1)
		curva.append(u+1) 
#Matriz de covarianza
	C = covarianza(sigma,len(simulado))
#Matriz de diseno para polinomio 3
	M=matriz_diseno(len(simulado),p)
	incertidumbre, parametros = ajuste(M, C, simulado, p)	
	Chi_cuadrado = chi_cuadrado(simulado,M,parametros,C)
	q = p_values(Chi_cuadrado,len(simulado),p+2)
	pvalues2.append(q)
print parametros
resultado = []
for i in range (0,len(time)):
	u = 0
	for t in range(0,len(parametros)):
		if (0.4 < time[i] < 0.7):
			if (t==0):
				u = u + parametros[t]
			else:
				u = u + parametros[t]*(time[i]**(t-1))
		else:
			if (t==0):
				u = 0
			else:
				u = u + parametros[t]*(time[i]**(t-1))
	resultado.append(u)
#Histograma de p-values ajustando
plt.hist(pvalues2, bins=20, normed=True, color = 'w', alpha=0.5)
plt.title('Histograma de p-values de 1000 simulaciones (p=5+transito)')
plt.xlabel('$\mathrm{Bin}$ $\mathrm{p-values}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'upper left')
plt.show()
plt.close()


#Plot de la ultima simulacion y del transito original
plt.errorbar(time, simulado, yerr=erro, fmt='o',color = 'b', label = 'Simulacion (funcion p=5 + transito)')
plt.plot(time,resultado, color = 'r', label = 'Curva de ajuste', lw=2)
plt.xlabel('$\mathrm{Tiempo[dias]}$')
plt.ylabel('$\mathrm{Flujo}$ $\mathrm{normalizado}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()
