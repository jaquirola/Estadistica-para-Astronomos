import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
import random

w = 0 #Numero de encuentas en la cual el numero de personas que NO cambian su opcion inicial es 19
vec = []
for k in range(1,1001): #Las 1000 encuestas que se va a simular
	y=0 #Contador: peronas que mantienen respuesta inicial
	z=0 #Contador: personas que cambian su respuesta inicial
	for i in range(1,34): #Las 33 personas encuestadas
		x= random.randint(0, 1) #La gente MANTIENE (0) o CAMBIA (1) su opcion inicial
		if x==0:
			y = y+1 
		else:
			z = z+1		
	vec.append(y)
	if y==18:
		w = w+1 
	#print k, y, z 
r = w/float(1000)
print "Numero de encuestas en las que hay 18 individuos que NO cambian su opcion inicial:", w, "de un total de 1000. La probabilidad de tener 18 es:", r 

plt.hist(vec,1000, (0,1000), label = 'Simulacion de 1000 encuestas')
plt.xlabel('$X$')
plt.ylabel('Frecuencia')
plt.title('Histograma de simulacion de 1000 encuestas')
#plt.legend(loc = 2)
plt.show()
