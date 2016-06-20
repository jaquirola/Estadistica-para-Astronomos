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
    if not 0 < rp < 1.0:
        return -np.inf
    if not 0 < a < 20:
        return -np.inf
    if not 0.0 < inc < 90.0:
        return -np.inf
    if not -1.0 < c < 1.0:
        return -np.inf
    return 0.0

#Prior del ruido

def lnprior_ind(p):
    sigma = p[0]
    if not 0 < sigma < 20:
        return -np.inf
    return lnprior_base(p[1:])

#Likelihood

def lnlike_ind(p, t, y):
    m = model(p[1:],t)
    return -0.5*np.sum(((y-m)/p[0])**2) + np.log(sqrt(1.0/(2*3.141691*p[0]**2)))

#Aposteriori

def lnprob_ind(p, t, y):
    lp = lnprior_ind(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_ind(p, t, y)

#Funcion de ajuste

def fit_ind(initial, data, nwalkers=20):
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
          for i in xrange(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_ind, args=data)

    print("Running burn-in")
    p0, _, _ = sampler.run_mcmc(p0, 1000)
    sampler.reset()

    print("Running production")
    p0, _, _ = sampler.run_mcmc(p0, 5000)
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

if __name__ == "__main__":
    t, y= datos[:,0],np.log(datos[:,1])
    truth = [0.0, 0.0, 0.5, 0.5, 50, 0.0]
    pl.plot(t, y,color='k',lw=2)
    pl.ylabel(r"Flujo")
    pl.xlabel(r"Tiempo")
    pl.xlim(-2.5, 2.5)
    pl.title("Estrella target")
    plt.show()

    data = (t, y)
    truth_ind = [4.0] + truth
    sampler = fit_ind(truth_ind, data)
     
#Modelo y ajustes
    samples = sampler.flatchain
    x = np.linspace(-2, 2, 100)
    for s in samples[np.random.randint(len(samples), size=20)]:
        pl.plot(x, model(s[1:], x), color="#4682b4", alpha=0.8)
    pl.plot(t, y,color="k", lw=2,label='Estrella target')
    pl.ylabel(r"Flujo")
    pl.xlabel(r"Tiempo")
    plt.legend(loc = 'lower left')
    plt.show()

#Aposteriori de cada parametro
    labels = [r"$\sigma$", r"$\alpha_1$", r"$\alpha_2$", r"$rp$", r"$a$", r"$inc$", r"$c$"]
    fig = triangle.corner(samples[:, 0:], labels=labels)
    plt.show()
