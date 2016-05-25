import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from scipy.stats import chisqprob, chi2
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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
#FUncion de la matriz de diseno
def matriz_diseno(dimension,p,a):
	M = np.zeros([dimension,p+2])
	for i in range (0,dimension,1):
		for j in range (0,p+2,1):
			if (a[i] < 0.4):
				if(j==0):
					M[i,j] = 0
				else:
					M[i,j] = (a[i])**(j-1) 
			elif (0.4 <= a[i] <= 0.7):
				if (j==0):
					M[i,j] = 1
				else:
					M[i,j] = (a[i])**(j-1)  
			else:
				if (j==0):
					M[i,j] = 0
				else:
					M[i,j] = (a[i])**(j-1) 
	return M
#Extraer los datos para las regiones de test y training
def regiones(k, test, ima, tiempo, N):
	for i in range (0,k):
		if (i != test):
			for j in range (0,N/k):
				x = ima[i][j]
				y = tiempo[i][j]
				ajustey.append(x)
				ajustex.append(y)
		else:
			for j in range (0,N/k):
				w = ima[i][j]
				z = tiempo[i][j]
				testy.append(w)
				testx.append(z)
	return ajustex, ajustey, testx, testy

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
p = 9
#Sigma de los datos 
sigma = 30e-6
flujo = []
for i in range(0,len(flux)):
	x = flux[i]-1
	flujo.append(x)

#Dividir el transito en k regiones
kfold = 10
ima = []
tiempo = []
#numero de region
for i in range (0,kfold):
	ima.append([])
	tiempo.append([])
	for j in range (i*(len(flujo)/kfold),(len(flujo)/kfold)*(i+1)):
		x = flujo[j]
		y = time[j]
#Guardar en una matriz el flujo
		ima[i].append(x)
#Guardar en una matriz el tiempo
		tiempo[i].append(y)
#Guarda un array de tiempo y flujo dividido por regiones
#RAngo

polinomio = []
MSE = []
for k in range (0,p+1):
	for h in range (0,kfold):
		ajustex = []
		ajustey = []
		testx= []
		testy = []
		suma = []
#Llamamos a definir las regiones de test y de training
		ajustex, ajustey, testx, testy = regiones(kfold, h, ima, tiempo, len(flujo))
#Matriz de incertidumbres
		C = covarianza(sigma,len(ajustey))
#Matriz de diseno
		M=matriz_diseno(len(ajustex), k, ajustex)
		incertidumbre, theta = ajuste(M, C, ajustey, k)
		resultado = []
#Valor evaluados
		for i in range (0,len(testx)):
			u = 0
			for t in range(0,len(theta)):
				if (0.4 < testx[i] < 0.7):
					if (t==0):
						u = u + theta[t]
					else:
						u = u + theta[t]*(testx[i]**(t-1))
				else:
					if (t==0):
						u = 0
					else:
						u = u + theta[t]*(testx[i]**(t-1))
			resultado.append(u)
		x = 0
		for i in range (0,len(testx)):		
			x = x + ((resultado[i]-testy[i])**2)/len(testx)
		suma.append(1e3*sqrt(x))
	promedio = 0
	for l in range (0,len(suma)):
		promedio = promedio + suma[l]/len(suma)
	MSE.append(promedio)
	polinomio.append(k)

#cross correlation

#plt.plot(polinomio, MSE, '-', color = 'b', lw = 3, label = 'K-fold Cross correlation')
#plt.legend(loc = 'upper left')
fig, ax = plt.subplots() # create a new figure with a default 111 subplot
plt.xlabel('$\mathrm{p}$')
plt.ylabel('$\mathrm{rms}$ $\mathrm{error}$ $\\times10^{3}$')
ax.plot(polinomio, MSE, lw = 3, label = 'K-fold Cross correlation', color = 'k')
axins = zoomed_inset_axes(ax, 2.5, loc=2) # zoom-factor: 2.5, location: upper-left
axins.plot(polinomio, MSE, lw = 3, color = 'k')
x1, x2, y1, y2 = 3.8, 5.2, 0.02, 0.05 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.legend(loc = 'upper right')
plt.show()
plt.close()
