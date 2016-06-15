import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from matplotlib.colors import ListedColormap


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
	

#Guardamos en diferentes arrays las clases
#1 y 2, para poder graficarlas como clases individuales

datos = loadtxt("datos_clasificacion.dat", float)
N = len(datos)
clase1 = []
clase2 = []
#Se guarda la clase1 y clase2 en diferentes arrays
for i in range (0,N):
	if (datos[i][2]==1):
		clase1.append([datos[i][0],datos[i][1],1])
	else:
		clase2.append([datos[i][0],datos[i][1],2]) 
#Se guarda los eementos de x1 y x2 de cada una de
# las clases
#Esto se usa para graficar los puntos
h = []
for i in range (0,len(clase1)):
	a = clase1[i][0]
	b = clase1[i][1]
	h.append([a,b])
class1 = np.squeeze(np.asarray(h))
h = []
for j in range (0,len(clase2)):
	a = clase2[j][0]
	b = clase2[j][1]
	h.append([a,b])
class2 = np.squeeze(np.asarray(h))

###############################################
#Definiendo probabilidad
##############################################

sigma1 = np.matrix([[5,-2],[-2,5]])
sigma2 = np.matrix([[1,0],[0,1]])
mu1 = np.matrix([2,3])
mu2 = np.matrix([6,6])

mu1 = [2, 3]
sigma1 = [[5, -2], [-2, 5]]  # diagonal covariance
mu2 = [6, 6]
sigma2 = [[1, 0], [0, 1]]  # diagonal covariance
X1,X2 = np.random.multivariate_normal(mu1, sigma1, 300).T
Y1,Y2 = np.random.multivariate_normal(mu2, sigma2, 300).T
plt.scatter(X1, X2,color='r',label = '$\mathrm{Clase 1}$')
plt.scatter(Y1, Y2,color='b',label = '$\mathrm{Clase 1}$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower left')
plt.ylim((-4,10))
plt.xlim((-4,11))
plt.show()



#######################
#numpy
####################
N1 = len(class1)/float(N)
N2 = len(class2)/float(N)
g1x = []
g1y = []
g2x = []
g2y = []
for x1 in my_range(-4,11,0.1):
	for x2 in my_range(-4,11,0.1):
		x = np.matrix([x1,x2])
		pG1 = log(N1)-0.5*log(np.linalg.det(sigma1))-0.5*((x-mu1)*inv(sigma1)*(x-mu1).T)
		pG2 = log(N2)-0.5*log(np.linalg.det(sigma2))-0.5*((x-mu2)*inv(sigma2)*(x-mu2).T)
		if (pG1>pG2):
			g1x.append(x1)
			g1y.append(x2)
		if (pG2>pG1):
			g2x.append(x1)
			g2y.append(x2)

plt.plot(g2x,g2y,'o',color='b',alpha=0.03,markersize=4)
plt.plot(g1x,g1y,'o',color='r',alpha=0.03,markersize=4)
plt.plot(class1[:,0], class1[:,1], '*',color='r', markersize=12, label = '$\mathrm{Clase 1}\ (\mathrm{Datos})$')
plt.plot(class2[:,0], class2[:,1], '*',color='b', markersize=12, label = '$\mathrm{Clase 2}\ (\mathrm{Datos})$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower left')
plt.ylim((-4,10))
plt.xlim((-4,11))
plt.show()

