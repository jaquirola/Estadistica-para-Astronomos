import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import numpy as np
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
y = np.squeeze(np.asarray(z))

n_neighbors = 15

# import some data to play with

h = .1  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.plot(clase1x,clase1y, '*',color='r', markersize=12, label = '$\mathrm{Clase 1}$')
plt.plot(clase2x,clase2y, '*',color='b', markersize=12, label = '$\mathrm{Clase 2}$')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc = 'lower left')
plt.savefig('neighbours.eps')
