import batman
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
import emcee
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import scipy.stats as stats
import emcee

def covarianza(sigma, r):
	C = np.zeros([r,r])
	for i in range (0,r,1):
		for j in range (0,r,1):
			if (i == j):
				C[i,j] = sigma**2
	return C

def ajuste (M,C,Y,p):
	theta = np.zeros([p+2,1])
	MT = M.transpose()
	CI = inv(C)
	prev = np.dot(MT,CI)
	incertidumbre = inv(np.dot(prev,M))
	prev2 =np.dot(prev,Y)
	theta = np.dot(incertidumbre, prev2)
	return incertidumbre, theta


datos = loadtxt("dataset_10.dat", float)

x = []
t = []
flujo = []
for i in range(0,len(datos)):
	if (-1<datos[i][0]<1):
		t.append(datos[i][0])
		flujo.append(log(datos[i][1]))
		x.append([log(datos[i][2]),log(datos[i][3]),log(datos[i][4]),log(datos[i][5]),log(datos[i][6]),log(datos[i][7]),log(datos[i][8]),log(datos[i][9]),log(datos[i][10])])
t = np.squeeze(np.asarray(t))
X = np.squeeze(np.asarray(x))
cov = np.dot(X.T,X)/9
values, vectors = linalg.eig(cov)
#Sort
idx = values.argsort()[::-1]   
eig_vals_sorted = values[idx]
eig_vecs_sorted = vectors[:,idx]
total = np.sum(eig_vals_sorted)
#Z=X.VT
Z = np.dot(X,eig_vecs_sorted)
Z = np.squeeze(np.asarray(Z))
F = np.squeeze(np.asarray(flujo))

y=10e-4



M = np.zeros([len(F),3])
for i in range (0,len(F)):
	for j in range (0,3):
		if (j==0):
			M[i,j] = 1
		if (j==1):
			M[i,j] = Z[i][0]
		if (j==2):
			M[i,j] = Z[i][1]
print M
C = covarianza(y,len(F))
theta = np.zeros([3,1])
MT = M.transpose()
CI = inv(C)
prev = np.dot(MT,CI)
incertidumbre = inv(np.dot(prev,M))
prev2 =np.dot(prev,F)
theta = np.dot(incertidumbre, prev2)
print theta











