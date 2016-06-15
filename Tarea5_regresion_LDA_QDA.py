import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D


#Funcion para obtener los parametros de ajuste theta y las incertidumbres
	
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def matrix(N,X):
	M = np.zeros((N,3))
	for i in range (0,N):
		for j in range (0,3):
			if (j==0):
				M[i][j]=1
			if (j==1):
				M[i][j] = X[i][0]
			if (j==2):
				M[i][j] = X[i][1]
	return M

def regresion (M,Y,p):
	theta = np.zeros([p,1])
	MT = M.transpose()
	prev = inv(np.dot(MT,M))
	prev2 =np.dot(MT,Y)
	theta = np.dot(prev, prev2)
	return theta

def lda(class1,class2,N):
	N1 = len(class1)/float(N)
	N2 = len(class2)/float(N) 
	mu1 = np.matrix([np.sum(class1[:,0])/len(class1),np.sum(class1[:,1])/len(class1)])
	mu2 = np.matrix([np.sum(class2[:,0])/len(class2),np.sum(class2[:,1])/len(class2)])
	sigma1 = np.zeros(shape=(2,2))
	for i in range(0,len(class1)):
		a = clase1[i][0]-mu1[0,0]
		b = clase1[i][1]-mu1[0,1]
		X = np.matrix([a,b])
		z = np.dot(X.T,X)
		sigma1 = sigma1 + z
	sigma1 = sigma1/(N-2)
	sigma2 = np.zeros(shape=(2,2))
	for i in range(0,len(class2)):
		a = class2[i][0]-mu2[0,0]
		b = class2[i][1]-mu2[0,1]
		X = np.matrix([a,b])
		z = np.dot(X.T,X)
		sigma2 = sigma2 + z
	sigma2 = sigma2/(N-2)
	sigma = (sigma1 + sigma2)
	a0 = log(N1/N2)-0.5*((mu1+mu2)*inv(sigma)*((mu1-mu2).T))
	A0 = np.squeeze(np.asarray(a0))
	a = inv(sigma)*((mu1-mu2).T)
	A = np.squeeze(np.asarray(a))
	return A0, A, sigma1, sigma2, mu1, mu2

def qda(class1, class2, N):
	N1 = len(class1)/float(N)
	N2 = len(class2)/float(N) 
	mu1 = np.matrix([np.sum(class1[:,0])/len(class1),np.sum(class1[:,1])/len(class1)])
	mu2 = np.matrix([np.sum(class2[:,0])/len(class2),np.sum(class2[:,1])/len(class2)])
	sigma1 = np.zeros(shape=(2,2))
	for i in range(0,len(class1)):
		a = class1[i][0]-mu1[0,0]
		b = class1[i][1]-mu1[0,1]
		X = np.matrix([a,b])
		z = np.dot(X.T,X)
		sigma1 = sigma1 + z
	sigma1 = sigma1/(N-2)
	sigma2 = np.zeros(shape=(2,2))
	for i in range(0,len(class2)):
		a = class2[i][0]-mu2[0,0]
		b = class2[i][1]-mu2[0,1]
		X = np.matrix([a,b])
		z = np.dot(X.T,X)
		sigma2 = sigma2 + z
	sigma2 = sigma2/(N-2)
	A0 = -0.5*(mu1*inv(sigma1)*mu1.T-mu2*inv(sigma2)*mu2.T)+log(N1/N2)-0.5*log(np.linalg.det(sigma1)/np.linalg.det(sigma2))	
	A1 = inv(sigma1)*mu1.T-inv(sigma2)*mu2.T
	A2 = -0.5*(inv(sigma1)-inv(sigma2))
	a0 = np.squeeze(np.asarray(A0))
	a1 = np.squeeze(np.asarray(A1))
	a2 = np.squeeze(np.asarray(A2))
	return a0, a1, a2

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


#Extraer los datos de un archivo .dat
############################################3
#Regresion

datos = loadtxt("datos_clasificacion.dat", float)
N = len(datos)
z = []
x = []
#Se guarda la clase1 y clase2 en diferentes arrays
for i in range (0,N):
	a = datos[i][0]
	b = datos[i][1]
	x.append([a,b])
	z.append(datos[i][2]) 
X = np.squeeze(np.asarray(x))
Y = np.squeeze(np.asarray(z))

#Se calcula la clasificacion usando el metodo de regresion lineal
#Y=beta0+x1beta1+x2beta2
beta = regresion(matrix(N,X),Y,3)
#Colocar puntos en las regiones clasificadas para diferenciar los puntos predichos a ser

R1 = []
R2 = []
for x1 in my_range(-4,12,0.2):
	R2.append(((1.5-beta[0]-beta[1]*x1)/beta[2]))
	R1.append(x1)
	
#Se grafica los datos clasificados con
#el metodo de regresion lineal

#Graficamos los puntos de cada clase
plt.plot(class1[:,0], class1[:,1], '*',color='r', markersize=12, label = '$\mathrm{Clase 1}$')
plt.plot(class2[:,0], class2[:,1], '*',color='b', markersize=12, label = '$\mathrm{Clase 2}$')
#Graficamos la funcion
plt.plot(R1,R2, lw=2, color = 'k', label = '$\mathrm{Regresion}$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.ylim((-4,10))
plt.legend(loc = 'lower left')
plt.show()
###########################################################################
#LDA

Y1 = []
Y2 = []
A0, A, sigma1, sigma2, mu1, mu2 = lda(class1, class2, N)
for x1 in my_range(-4,12,0.5):
	Y2.append((-A0-A[0]*x1)/A[1])
	Y1.append(x1)

#Se grafica los datos clasificados con
#el metodo LDA

#Graficamos los puntos de cada clase
plt.plot(class1[:,0], class1[:,1], '*',color='r', markersize=12, label = '$\mathrm{Clase 1}\ \mathrm{Datos}$')
plt.plot(class2[:,0], class2[:,1], '*',color='b', markersize=12, label = '$\mathrm{Clase 2}\ \mathrm{Datos}$')
plt.plot(Y1,Y2, lw=3, color = 'k', label = '$\mathrm{LDA}$')
#plt.plot(R1,R2, '--', lw=3, color = 'c', label = '$\mathrm{Regresion}$')
plt.legend(loc = 'lower left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.ylim((-5,10))
plt.xlim((-6,11))
plt.show()

##########################################################
#Se realiza el quadratic discrimination analysis
#QDA

a0, a1,a2 = qda(class1, class2, N)
print a0
print a1
print a2
#Se tiene la ecuacion a0+a1x1+a1'x2+a2x1+a2'x2^2+a3x1x2
X1 = []
X2 = []
X3 = []
for x1 in my_range(2.639,12,0.01):
	a = a2[1][1]
	b = a1[1]+2*a2[0][1]*x1
	c = a0+a1[0]*x1+a2[0][0]*x1**2
	z1 = (-b+sqrt(b**2-4*a*c))/(2*a)
	z2 = (-b-sqrt(b**2-4*a*c))/(2*a)
	X2.append(z1)
	X1.append(x1)
	X3.append(z2)

#Se grafica los datos clasificados con el metodo QDA

plt.plot(class1[:,0], class1[:,1], '*',color='r', markersize=12, label = '$\mathrm{Clase 1}\ \mathrm{Datos}$')
plt.plot(class2[:,0], class2[:,1], '*',color='b', markersize=12, label = '$\mathrm{Clase 2}\ \mathrm{Datos}$')
plt.plot(X1, X2, color = 'k', lw=3,  label = '$\mathrm{QDA}$')
plt.plot(X1, X3, color = 'k', lw=3)
plt.plot(Y1, Y2, lw=3, color = 'k', label = '$\mathrm{LDA}$')
plt.plot(R1,R2, '--', lw=3, color = 'm', label = '$\mathrm{Regresion}$')
plt.ylim((-5,10))
plt.xlim((-6,11))
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower left')
plt.show()

