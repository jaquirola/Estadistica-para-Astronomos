import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
from numpy import *
import matplotlib.pyplot as plt
from math import *

#Guardar por clase

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


clase1x = []
clase1y = []
for i in range (0,len(clase1)):
	clase1x.append(clase1[i][0])
	clase1y.append(clase1[i][1])
clase2x = []
clase2y = []
for j in range (0,len(clase2)):
	clase2x.append(clase2[j][0])
	clase2y.append(clase2[j][1])

#Guardar indistintamente la clase

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


# fit the model
clf = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200, fit_intercept=True)
clf.fit(X, Y)

# plot the line, the points, and the nearest vectors to the plane
xx = np.linspace(-4, 11, 10)
yy = np.linspace(-4, 11, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles, lw = 3)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.plot(clase1x,clase1y, '*',color='r', markersize=12, label = '$\mathrm{Clase 1}$')
plt.plot(clase2x,clase2y, '*',color='b', markersize=12, label = '$\mathrm{Clase 2}$')
plt.axis('tight')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower left')
plt.show()
