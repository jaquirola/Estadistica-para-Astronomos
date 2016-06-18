import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from sklearn.decomposition import PCA

datos = loadtxt("dataset_10.dat", float)

x = []
for i in range(0,len(datos)):
	x.append([log(datos[i][2]),log(datos[i][3]),log(datos[i][4]),log(datos[i][5]),log(datos[i][6]),log(datos[i][7]),log(datos[i][8]),log(datos[i][9]),log(datos[i][10])])

X = np.squeeze(np.asarray(x))
cov = np.cov(X.T)
values, vectors = linalg.eig(cov)
#Sort
eig_vals_sorted = np.sort(values)
eig_vecs_sorted = vectors[:, values.argsort()]
total = np.sum(eig_vals_sorted)
#Z=X.VT
Z = np.dot(X,eig_vecs_sorted)
num = []
fig, ax = plt.subplots(ncols=3,nrows=3,sharex=True,figsize=[4*3,2.5*4])
j=k=0
for i in range(0,9):
	ax[j,k].plot(datos[:,0],Z[:,i],lw=2,color='k')
	num.append(i)
	plt.legend(loc = 'lower left')
	fig.tight_layout()
	k = k+1
    	if (k > 2):
        	k=0
		j=j+1
#plt.xlabel('$\mathrm{Tiempo[horas]}$')
#plt.ylabel('$\mathrm{Flujo}$')
plt.show()


importantes = eig_vals_sorted/total
plt.plot(num,importantes,lw=2,color='k')
plt.xlabel('$\mathrm{P}$')
plt.ylabel('$\%$')
plt.legend(loc = 'lower left')
plt.show()
