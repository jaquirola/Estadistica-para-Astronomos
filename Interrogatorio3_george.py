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
import numpy.random as nr

def transito(rp,a,inc,t):
	params = batman.TransitParams()
	params.t0 = 0.                       #time of inferior conjunction
	params.per = 0.78884                      #orbital period
	params.rp = rp                      #planet radius (in units of stellar radii)
	params.a = a                       #semi-major axis (in units of stellar radii)
	params.inc = inc                     #orbital inclination (in degrees)
	params.ecc = 0.                      #eccentricity
	params.w = 90.                       #longitude of periastron (in degrees)
	params.u = [0.1, 0.3]                #limb darkening coefficients
	params.limb_dark = "quadratic"       #limb darkening model
	m = batman.TransitModel(params, t)    #initializes model
	flux = m.light_curve(params)          #calculates light curve
	return flux

def modelo(parametros,t):
	alpha1, alpha2, rp, a, inc, c = parametros
	T = np.log(transito(rp,a,inc,t))
	mod = c + T + alpha1*Z[:,0] + alpha2*Z[:,1]
	return mod


def log_likelihood(parametros,t,F,y):
	alpha1, alpha2, rp, a, inc, c = parametros
	residuos = (F-modelo(parametros,t))**2/y**2
	likelihood = np.sum(-0.5*residuos)+log(1./sqrt(2.*3.141691*y**2))
	return likelihood

def log_prior(p):
    	alpha1, alpha2, rp, a, inc, c = p
    	if (-1 < alpha1 < 1):
		prior_alpha1 = 0.5	
	else:
		prior_alpha1 = 0	
	if (-1 < alpha2 < 1):
		prior_alpha2 = 0.5
	else:
		prior_alpha2 = 0
	if (0 < rp < 1):
		prior_rp = 1/1 
	else:
		prior_rp = 0
	if (0 < a < 1):
		prior_a = 1/1 
	else:
		prior_a = 0
	if (-1 < inc < 1):
		prior_inc = 0.5
	else:
		prior_inc = 0
	if (-1 < c < 1):
        	prior_c= 0.5
	else:
		prior_c=0
	prior=prior_alpha1+prior_alpha2+prior_rp+prior_a+prior_inc+prior_c
   	return prior
   

def log_posterior(parametros,t,F,y):
   	lp = log_prior(parametros)
   	aposteriori = lp+log_likelihood(parametros,t,F,y)
    	return aposteriori 

def metropolis(log_posterior, parametros, stepsize, nsteps):
    alpha1, alpha2, rp, a, inc, c = parametros
    chain = np.zeros((nsteps, 6))
    acceptance = 0
    log_probs = []
    parametros1 = parametros
#    parametros1 = parametros
    for i in range(0,nsteps):
        u = np.random.rand()
	parametros1 = nr.multivariate_normal(parametros, np.diag(stepsize), 1)
	parametros1=np.squeeze(np.asarray(parametros1))
        a = log_posterior(parametros1,t,F,y)
	b = log_posterior(parametros,t,F,y)
	c = a/b
        alpha = np.amin([c,1])
        if (alpha>u):
            parametros = parametros1
            chain[i] = parametros1
            log_probs.append(log_posterior(parametros,t,F,y))
            acceptance = acceptance + 1
        else:
            chain[i] = parametros
            log_probs.append(log_posterior(parametros,t,F,y))
#            acceptance = acceptance + 1           
    acceptance_rate = acceptance/float(nsteps)
    return chain,log_probs,acceptance_rate


datos = loadtxt("dataset_10.dat", float)

x = []
t = []
flujo = []
for i in range(0,len(datos)):
	t.append(datos[i][0])
	flujo.append(log(datos[i][1]))
	x.append([log(datos[i][2]),log(datos[i][3]),log(datos[i][4]),log(datos[i][5]),log(datos[i][6]),log(datos[i][7]),log(datos[i][8]),log(datos[i][9]),log(datos[i][10])])
t = np.squeeze(np.asarray(t))
X = np.squeeze(np.asarray(x))
cov = np.dot(X.T,X)/100
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

y = np.mean(datos[:,1])
suma =0
for i in range(0,len(datos)):
	suma = suma + (datos[i][1]-y)**2
y = suma/len(datos)
print y
#y=10e-4
'''
data = (t,F,y)

nwalkers = 18
initial = np.array([0, 0, 10, 0.5, 0.5, 0, 0])
ndim = len(initial)
p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=data)

print("Running burn-in...")
p0, _, _ = sampler.run_mcmc(p0, 200)
sampler.reset()

print("Running production...")
sampler.run_mcmc(p0, 500)

'''
alpha1step = 0.003
alpha2step = 0.003
#sigmastep = 0.03
rpstep = 0.003
astep = 0.003
incstep = 0.003
cstep = 0.003
stepsize = np.array([alpha1step,alpha2step,rpstep,astep,incstep,cstep]) 
alpha1=1
alpha2=1
#sigma=0.5
rp=0.1
a=0.1
inc=1
c=0.5
parametros = [alpha1, alpha2,rp, a, inc, c]
#print parametros
#print t
#print F
#print y

nsteps=10000
cadena, log_probs, acceptance_rate = metropolis(log_posterior, parametros, stepsize, nsteps)
plt.plot(cadena[:,3])
plt.show()
chain = []
burn_in = 100
for i in range(0,len(cadena)):
	if (i>burn_in):
		chain.append(cadena[i])
chain = np.squeeze(np.asarray(chain))
print chain
promedios = [np.mean(chain[:,0]),np.mean(chain[:,1]),np.mean(chain[:,2]),np.mean(chain[:,3]),np.mean(chain[:,4]),np.mean(chain[:,5])]
promedios = np.squeeze(np.asarray(promedios))
#constantes = modelo(promedios,t)
plt.hist(chain[:,0], bins=30, normed=False, color = 'k',label='$\\alpha_1$')
plt.axvline(x=promedios[0], lw = float(2.0), ls = 'dashed', color ='c')
plt.xlabel('$\mathrm{Bin}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

plt.hist(chain[:,1], bins=30, normed=False, color = 'k',label='$\\alpha_2$')
plt.axvline(x=promedios[1], lw = float(2.0), ls = 'dashed', color ='c')
plt.xlabel('$\mathrm{Bin}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

plt.hist(chain[:,2], bins=30, normed=False, color = 'k',label='$R_p/R*$')
plt.axvline(x=promedios[2], lw = float(2.0), ls = 'dashed', color ='c')
plt.xlabel('$\mathrm{Bin}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

plt.hist(chain[:,3], bins=30, normed=False, color = 'k',label='$a/R*$')
plt.axvline(x=promedios[3], lw = float(2.0), ls = 'dashed', color ='c')
plt.xlabel('$\mathrm{Bin}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

plt.hist(chain[:,4], bins=30, normed=False, color = 'k',label='$i$')
plt.axvline(x=promedios[4], lw = float(2.0), ls = 'dashed', color ='c')
plt.xlabel('$\mathrm{Bin}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

plt.hist(chain[:,5], bins=30, normed=False, color = 'k',label='$c$')
plt.axvline(x=promedios[5], lw = float(2.0), ls = 'dashed', color ='c')
plt.xlabel('$\mathrm{Bin}$')
plt.ylabel('$\mathrm{Frecuencia}$')
plt.legend(loc = 'lower left')
plt.show()
plt.close()

















