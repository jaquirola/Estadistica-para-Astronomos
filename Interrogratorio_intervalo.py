from __future__ import division, print_function
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl
import batman
import george
from george import kernels
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.68):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


#Batman
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
	m = batman.TransitModel(params, t/24)    #initializes model
	flux = m.light_curve(params)          #calculates light curve
	return flux


#Modelo
def model(params, t):
    alpha1, alpha2, rp, a, inc, c = params 
    T = transito(rp,a,inc,t/24)
    return c + np.log(T) + alpha1*Z[:,0] + alpha2*Z[:,1]

#Prior

def lnprior_base(p):
    alpha1, alpha2, rp, a, inc, c = p
    if not -1.0 < alpha1 < 1.0:
        return -np.inf
    if not -1.0 < alpha2 < 1.0:
        return -np.inf
    if not 0 < rp < .3:
        return -np.inf
    if not 0 < a < 10.0:
        return -np.inf
    if not 0.0 < inc < 90.0:
        return -np.inf
    if not -1.0 < c < 1.0:
        return -np.inf
    return 0.0

#likelihood proceso gaussiano

def lnlike_gp(p, t, y):
    l = np.exp(p[0])
    gp = george.GP(kernels.ExpSquaredKernel(l))
    gp.compute(t, p[1])
    return gp.lnlikelihood(y - model(p[2:], t))

#Prior de los parmaetros del proceso gaussiano

def lnprior_gp(p):
    l, sigma = p[:2]
    if not -10 < l < 10:
        return -np.inf
    if not 0 < sigma < 10:
        return -np.inf
    return lnprior_base(p[2:])

#Aposteriori

def lnprob_gp(p, t, y):
    lp = lnprior_gp(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_gp(p, t, y)

#Funcion del fit gaussiano

def fit_gp(initial, data, nwalkers=32):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_gp, args=data)

    print("Running burn-in")
    p0, lnp, _ = sampler.run_mcmc(p0, 700)
    sampler.reset()

    print("Running second burn-in")
    p = p0[np.argmax(lnp)]
    p0 = [p + 1e-8 * np.random.randn(ndim) for i in xrange(nwalkers)]
    p0, _, _ = sampler.run_mcmc(p0, 500)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 1000)

    return sampler

#Extraer datos


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


t, y= datos[:,0],np.log(datos[:,1])
truth = [0.0, 0.0, 0.2, 0.5, 50, 0.0]

data = (t, y)
truth_gp = [0.0,5.0] + truth
sampler = fit_gp(truth_gp, data)

a = sampler.flatchain[:,4]
q,y,z = mean_confidence_interval(a, confidence=0.95)
b = median(a)

print ('Mediana:'+str(b))
print ('[a:'+str(b-y))
print ('b]:'+str(z-b))
