import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs
from sklearn import linear_model


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
k = 10
#Se guarda la clase1 y clase2 en diferentes arrays
#numero de region

################################################################
#Metodo regresion
################################################################
training = []
training1 = []
test = []
clas1clas1 = 0
clas1clas2 = 0
clas2clas2 = 0
clas2clas1 = 0
for i in range (0,k):
	training.append([])
	training1.append([])
	test.append([])
	for j in range (0,N):
		if (j>=i*(N/k) and j<(N/k)*(i+1)):
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			test.append([a,b,c])
		else:
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			training.append([a,b,c])
			training1.append([a,b])
	list2 = filter(None, test)
	prueba = np.squeeze(np.asarray(list2))
	list2 = filter(None, training)
	entrenamiento = np.squeeze(np.asarray(list2))
	list2 = filter(None, training1)
	entrenamiento1 = np.squeeze(np.asarray(list2))
	beta = regresion(matrix(len(entrenamiento1),entrenamiento1),entrenamiento[:,2],3)
	#Numero de aciertos y desaciertos de clase 1
	truec1 = 0
	falsec1 = 0
	#Numero de aciertos y desaciertos de clase 2
	truec2 = 0
	falsec2 = 0
	for l in range(0, len(prueba)):
		y = beta[0]+beta[1]*prueba[l][0]+beta[2]*prueba[l][1]
	#Son clase 1 y predigo que son clase 1
		if (y <= 1.5 and prueba[l][2]==1):
			truec1 = truec1 + 1
	#Son clase 2 y predigo que son clase 1
		if (y <= 1.5 and prueba[l][2]==2):
			falsec1 = falsec1 + 1
	#Son clase 2 y predigo que son clase 2
		if (y > 1.5 and prueba[l][2]==2):
			truec2 = truec2 + 1
	#Son clase 1 y predigo que son clase 2
		if (y > 1.5 and prueba[l][2]==1):
			falsec2 = falsec2 + 1
	clas1clas1 = clas1clas1 + truec1
	clas1clas2 = clas1clas2 + falsec1
	clas2clas2 = clas2clas2 + truec2
	clas2clas1 = clas2clas1 + falsec2	
	training = []
	test = []
	training1 = []
print 'Regresion' 
print 'Predigo1/real1:',clas1clas1, clas1clas1/float(N)
print 'Predigo1/real2:',clas1clas2, clas1clas2/float(N)
print 'Predigo2/real2:',clas2clas2, clas2clas2/float(N)
print 'Predigo2/real1:',clas2clas1, clas2clas1/float(N)
print 'Misclassification rate', 1-((clas1clas1+clas2clas2)/float(N))


############################################3
#Metodo-LDA
###########################################3

training = []
test = []
clas1clas1 = 0
clas1clas2 = 0
clas2clas2 = 0
clas2clas1 = 0
for i in range (0,k):
	training.append([])
	test.append([])
	for j in range (0,N):
		if (j>=i*(N/k) and j<(N/k)*(i+1)):
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			test.append([a,b,c])
		else:
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			training.append([a,b,c])
	list2 = filter(None, test)
	prueba = np.squeeze(np.asarray(list2))
	list2 = filter(None, training)
	entrenamiento = np.squeeze(np.asarray(list2))
	h = []
	g = []
	for j in range (0,len(entrenamiento)):
		if (entrenamiento[j][2]==1):
			a = entrenamiento[j][0]
			b = entrenamiento[j][1]
			h.append([a,b])
		else:
			a = entrenamiento[j][0]
			b = entrenamiento[j][1]
			g.append([a,b])
	class1 = np.squeeze(np.asarray(h))
	class2 = np.squeeze(np.asarray(g))		
	Y1 = []
	Y2 = []
#Aplicamos nuestro metodo LDA
	A0, A, sigma1, sigma2, mu1, mu2 = lda(class1, class2, len(entrenamiento))
	#Numero de aciertos y desaciertos de clase 1
	truec1 = 0
	falsec1 = 0
	#Numero de aciertos y desaciertos de clase 2
	truec2 = 0
	falsec2 = 0
	for l in range(0, len(prueba)):
		y = A0+A[0]*prueba[l][0]+A[1]*prueba[l][1]
	#Son clase 1 y predigo que son clase 1
		if (y >= 0 and prueba[l][2]==1):
			truec1 = truec1 + 1
	#Son clase 2 y predigo que son clase 1
		if (y >= 0 and prueba[l][2]==2):
			falsec1 = falsec1 + 1
	#Son clase 2 y predigo que son clase 2
		if (y < 0 and prueba[l][2]==2):
			truec2 = truec2 + 1
	#Son clase 1 y predigo que son clase 2
		if (y < 0 and prueba[l][2]==1):
			falsec2 = falsec2 + 1
	clas1clas1 = clas1clas1 + truec1
	clas1clas2 = clas1clas2 + falsec1
	clas2clas2 = clas2clas2 + truec2
	clas2clas1 = clas2clas1 + falsec2
	training = []
	test = []
print 'LDA' 
print 'Predigo1/real1:',clas1clas1, clas1clas1/float(N)
print 'Predigo1/real2:',clas1clas2, clas1clas2/float(N)
print 'Predigo2/real2:',clas2clas2, clas2clas2/float(N)
print 'Predigo2/real1:',clas2clas1, clas2clas1/float(N)
print 'Misclassification rate', 1-((clas1clas1+clas2clas2)/float(N))
############################################3
#Metodo-QDA
###########################################3

training = []
test = []
clas1clas1 = 0
clas1clas2 = 0
clas2clas2 = 0
clas2clas1 = 0
for i in range (0,k):
	training.append([])
	test.append([])
	for j in range (0,N):
		if (j>=i*(N/k) and j<(N/k)*(i+1)):
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			test.append([a,b,c])
		else:
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			training.append([a,b,c])
	list2 = filter(None, test)
	prueba = np.squeeze(np.asarray(list2))
	list2 = filter(None, training)
	entrenamiento = np.squeeze(np.asarray(list2))
	h = []
	g = []
	for j in range (0,len(entrenamiento)):
		if (entrenamiento[j][2]==1):
			a = entrenamiento[j][0]
			b = entrenamiento[j][1]
			h.append([a,b])
		else:
			a = entrenamiento[j][0]
			b = entrenamiento[j][1]
			g.append([a,b])
	class1 = np.squeeze(np.asarray(h))
	class2 = np.squeeze(np.asarray(g))	
#Aplicamos nuestro metodo QDA	
	a0, a1,a2 = qda(class1, class2, len(entrenamiento))
#Se tiene la ecuacion a0+a1x1+a1'x2+a2x1+a2'x2^2+a3x1x2
	#Numero de aciertos y desaciertos de clase 1
	truec1 = 0
	falsec1 = 0
	#Numero de aciertos y desaciertos de clase 2
	truec2 = 0
	falsec2 = 0
	for l in range(0, len(prueba)):
		y = a0+a1[0]*prueba[l][0]+a1[1]*prueba[l][1]+a2[0][0]*(prueba[l][0]**2)+a2[1][1]*(prueba[l][1]**2)+2*a2[0][1]*(prueba[l][0]*prueba[l][1])
	#Son clase 1 y predigo que son clase 1
		if (y > 0 and prueba[l][2]==1):
			truec1 = truec1 + 1
	#Son clase 2 y predigo que son clase 1
		if (y > 0 and prueba[l][2]==2):
			falsec1 = falsec1 + 1
	#Son clase 2 y predigo que son clase 2
		if (y < 0 and prueba[l][2]==2):
			truec2 = truec2 + 1
	#Son clase 1 y predigo que son clase 2
		if (y < 0 and prueba[l][2]==1):
			falsec2 = falsec2 + 1
	clas1clas1 = clas1clas1 + truec1
	clas1clas2 = clas1clas2 + falsec1
	clas2clas2 = clas2clas2 + truec2
	clas2clas1 = clas2clas1 + falsec2
	X1 = []
	X2 = []
	X3 = []
	for x1 in my_range(3.574,12,0.01):
		a = a2[1][1]
		b = a1[1]+a2[0][1]*x1
		c = a0+a1[0]*x1+a2[0][0]*x1**2
		z1 = (-b+sqrt(abs(b**2-4*a*c)))/(2*a)
		z2 = (-b-sqrt(abs(b**2-4*a*c)))/(2*a)
		X2.append(z1)
		X1.append(x1)
		X3.append(z2)
	training = []
	test = []
print 'QDA'
print 'Predigo1/real1:',clas1clas1, clas1clas1/float(N)
print 'Predigo1/real2:',clas1clas2, clas1clas2/float(N)
print 'Predigo2/real2:',clas2clas2, clas2clas2/float(N)
print 'Predigo2/real1:',clas2clas1, clas2clas1/float(N)
print 'Misclassification rate', 1-((clas1clas1+clas2clas2)/float(N))

#######################################################3
#Nearest neightbors
#################################################333
training = []
test = []
clas1clas1 = 0
clas1clas2 = 0
clas2clas2 = 0
clas2clas1 = 0
for i in range (0,k):
	
	training.append([])
	test.append([])
	for j in range (0,N):
		if (j>=i*(N/k) and j<(N/k)*(i+1)):
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			test.append([a,b,c])
		else:
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			training.append([a,b,c])
	list2 = filter(None, test)
	prueba = np.squeeze(np.asarray(list2))
	list2 = filter(None, training)
	entrenamiento = np.squeeze(np.asarray(list2))
	n_neighbors = 15
	#Probabilidad uniforme
	clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
	clf.fit(entrenamiento[:, [0, 1]], entrenamiento[:,2])
	#Predecimos para cada valor de prueba
	Z = clf.predict(np.c_[prueba[:,0].ravel(), prueba[:,1].ravel()])
	training = []
	test = []
	#Numero de aciertos y desaciertos de clase 1
	truec1 = 0
	falsec1 = 0
	#Numero de aciertos y desaciertos de clase 2
	truec2 = 0
	falsec2 = 0
	for l in range(0, len(Z)):
	#Son clase 1 y predigo que son clase 1
		if (Z[l]==1.0 and prueba[l][2]==1):
			truec1 = truec1 + 1
	#Son clase 2 y predigo que son clase 1
		if (Z[l]==1.0 and prueba[l][2]==2):
			falsec1 = falsec1 + 1
	#Son clase 2 y predigo que son clase 2
		if (Z[l]==2.0 and prueba[l][2]==2):
			truec2 = truec2 + 1
	#Son clase 1 y predigo que son clase 2
		if (Z[l]==2.0 and prueba[l][2]==1):
			falsec2 = falsec2 + 1
	clas1clas1 = clas1clas1 + truec1
	clas1clas2 = clas1clas2 + falsec1
	clas2clas2 = clas2clas2 + truec2
	clas2clas1 = clas2clas1 + falsec2
#	preficcion.np.array([clas1clas1+clas1clas1,clas1clas2+clas1clas2,clas2clas2+clas2clas2,])
print 'Nearest neighbors'
print 'Predigo1/real1:',clas1clas1, clas1clas1/float(N)
print 'Predigo1/real2:',clas1clas2, clas1clas2/float(N)
print 'Predigo2/real2:',clas2clas2, clas2clas2/float(N)
print 'Predigo2/real1:',clas2clas1, clas2clas1/float(N)
print 'Misclassification rate', 1-((clas1clas1+clas2clas2)/float(N))

##################################################
#SGD
#################################################

training = []
test = []
clas1clas1 = 0
clas1clas2 = 0
clas2clas2 = 0
clas2clas1 = 0
for i in range (0,k):
	
	training.append([])
	test.append([])
	for j in range (0,N):
		if (j>=i*(N/k) and j<(N/k)*(i+1)):
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			test.append([a,b,c])
		else:
			a = datos[j][0]
			b = datos[j][1]
			c = datos[j][2]
			training.append([a,b,c])
	list2 = filter(None, test)
	prueba = np.squeeze(np.asarray(list2))
	list2 = filter(None, training)
	entrenamiento = np.squeeze(np.asarray(list2))
	clf = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200, fit_intercept=True)
	clf.fit(entrenamiento[:, [0, 1]], entrenamiento[:,2])
	Z = clf.predict(prueba[:, [0, 1]])
	training = []
	test = []
		#Numero de aciertos y desaciertos de clase 1
	truec1 = 0
	falsec1 = 0
	#Numero de aciertos y desaciertos de clase 2
	truec2 = 0
	falsec2 = 0
	for l in range(0, len(Z)):
	#Son clase 1 y predigo que son clase 1
		if (Z[l]==1.0 and prueba[l][2]==1):
			truec1 = truec1 + 1
	#Son clase 2 y predigo que son clase 1
		if (Z[l]==1.0 and prueba[l][2]==2):
			falsec1 = falsec1 + 1
	#Son clase 2 y predigo que son clase 2
		if (Z[l]==2.0 and prueba[l][2]==2):
			truec2 = truec2 + 1
	#Son clase 1 y predigo que son clase 2
		if (Z[l]==2.0 and prueba[l][2]==1):
			falsec2 = falsec2 + 1
	clas1clas1 = clas1clas1 + truec1
	clas1clas2 = clas1clas2 + falsec1
	clas2clas2 = clas2clas2 + truec2
	clas2clas1 = clas2clas1 + falsec2
#	preficcion.np.array([clas1clas1+clas1clas1,clas1clas2+clas1clas2,clas2clas2+clas2clas2,])
print 'SGD'
print 'Predigo1/real1:',clas1clas1, clas1clas1/float(N)
print 'Predigo1/real2:',clas1clas2, clas1clas2/float(N)
print 'Predigo2/real2:',clas2clas2, clas2clas2/float(N)
print 'Predigo2/real1:',clas2clas1, clas2clas1/float(N)
print 'Misclassification rate', 1-((clas1clas1+clas2clas2)/float(N))
