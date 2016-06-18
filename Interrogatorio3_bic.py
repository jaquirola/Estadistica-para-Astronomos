import batman
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from math import *
from numpy.linalg import inv
import emcee
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import scipy.stats as stats

#Modelo de batman
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

#Modelo
def modelo(promedios,t):
	alpha1, alpha2, alpha3, alpha4, sigmay, rp, a, inc, c = promedios
	T = np.squeeze(np.asarray(map(log,transito(rp,a,inc,t))))
	mod = c + T + alpha1*Z[:,8] + alpha2*Z[:,7] + alpha3*Z[:,6] + alpha4*Z[:,5]
	return mod
#Likelihood
def log_likelihood(Z,parametros,alpha,F,p):
	sigmay, rp, a, inc, c = parametros
#	print alpha
	if (rp>1):
		rp = 1.
	if (rp<0):
		rp = 0.1
	if (sigmay<0):
		sigmay=0.1
	if (a<0):
		a = 0.1
	if (a>1):
		a = 1
	if (inc<0):
		inc = 0.1
	T = np.squeeze(np.asarray(map(log,transito(rp,a,inc,t))))
	model = 0	
	for i in range(0,p):
		model = model + (c + T + alpha[i]*Z[:,i])
	likelihood = np.sum(log(1./sqrt(2.*3.141691)*sigmay))+np.sum((-0.5/sigmay**2)*(F-model)**2)
#	print parametros
	return likelihood
#Prior
def log_prior(parametros, alpha, limites, alphalim,p):
    sigmay, rp, a, inc, c = parametros
    sigmay_lim, rp_lim, a_lim, inc_lim, c_lim = limites

    # Uniform in sigmay:
    if (sigmay <  sigmay_lim[0]) | (sigmay >  sigmay_lim[1]):
        log_sigmay_prior = 0
    else:
        log_sigmay_prior = np.log(1.0/float(sigmay_lim[1] - sigmay_lim[0]))
    
    # Uniform in alpha:
    log_alpha_prior = 0
    for i in range(0,p):
	    a = 0
	    if (alpha[i] < alphalim[i][0]) | (alphalim[i] > alphalim[i][1]):
	        a = 0
    	    else:
        	a = np.log(1.0/float(alphalim[i][1] - alphalim[i][0]))
	    log_alpha_prior = log_alpha_prior + a
    # Uniform in rp:
    if (rp < rp_lim[0]) | (rp > rp_lim[1]):
        log_rp_prior = 0
    else:
        log_rp_prior = np.log(1.0/float(rp_lim[1] - rp_lim[0]))
    # Uniform in c:
    if (c < c_lim[0]) | (c > c_lim[1]):
        log_c_prior = 0
    else:
        log_c_prior = np.log(1.0/float(c_lim[1] - c_lim[0]))
    # Uniform in a:
    if (a < a_lim[0]) | (a > a_lim[1]):
        log_a_prior = 0
    else:
        log_a_prior = np.log(1.0/float(a_lim[1] - a_lim[0]))
    # Uniform in inc:
    if (inc < inc_lim[0]) | (inc > inc_lim[1]):
        log_inc_prior = 0
    else:
        log_inc_prior = np.log(1.0/(inc_lim[1] - inc_lim[0]))
    prior = log_sigmay_prior + log_alpha_prior + log_rp_prior + log_c_prior + log_a_prior + log_inc_prior
#    print prior
    return prior
#Posteriori
def log_posterior(Z,F,parametros,alpha,alphalim,limites,p):
    aposteriori = log_likelihood(Z,parametros,alpha,F,p)+log_prior(parametros, alpha, limites, alphalim, p)
    return aposteriori
#MCMC metropolis hasting
def metropolis(log_posterior, parametros, alpha, alphalim, limites, stepsize, alphastep, nsteps,p):
    chain = np.zeros((nsteps, len(parametros)+len(alpha)))
    acceptance = 0
    log_probs = []
    parametros1 = parametros
    alpha = np.squeeze(np.asarray(alpha))
    for i in range(0,nsteps):
        u = np.random.rand()
	parametros1 = []
	alpha1 = []
	for j in range(0,len(parametros)):
		parametros1.append(np.random.normal(parametros[j],stepsize[j]))
	for j in range(0,p):
		alpha1.append(np.random.normal(alpha[j],alphastep[j]))
	alpha1 = np.squeeze(np.asarray(alpha1))
        a = log_posterior(Z, F, parametros, alpha,alphalim,limites,p)/log_posterior(Z, F, parametros1, alpha1,alphalim, limites,p)
        if (np.amin([a,1])>u):
            parametros = parametros1
	    alpha = alpha1
            chain[i] = np.concatenate((parametros1,alpha1), axis=0)
            log_probs.append(log_posterior(Z, F, parametros, alpha, alphalim, limites,p))
            acceptance = acceptance + 1
        else:
            chain[i] = np.concatenate((parametros,alpha), axis=0)
            log_probs.append(log_posterior(Z, F, parametros, alpha, alphalim, limites, p))
            acceptance = acceptance + 1           
    acceptance_rate = acceptance/float(nsteps)
    return chain,log_probs,acceptance_rate
#Definir los limites y valores iniciales
#de los parametros
def par_lim(p):
	#Limites
	alphalim = []
	for i in range(0,p):
		alphalim.append([-20,20])  
	sigmaylim = [0.1,2]
	rplim = [0.1,0.9]
	alim = [0.1,10]
	inclim = [1,80]
	clim = [0,2]
	limites = (sigmaylim, rplim, alim, inclim, clim)
	#parametros
	alpha = []
	for i in range(0,p):
		alpha.append([0.5*(alphalim[i][0]+alphalim[i][1])])  
	sigmay = 0.5*(sigmaylim[0]+sigmaylim[1])
	rp = 0.5*(rplim[0]+rplim[1])
	a = 0.5*(alim[0]+alim[1])
	inc = 0.5*(inclim[0]+inclim[1])
	c = 0.5*(clim[0]+clim[1])
	parametros = [sigmay, rp, a, inc, c]
	# step sizes 
	alphastep = []
	for i in range(0,p):
		alphastep.append([0.1*(alphalim[i][1]-alphalim[i][0])])  
	sigmaystep = 0.1*(sigmaylim[1]-sigmaylim[0])
	rpstep = 0.1*(rplim[1]-rplim[0])
	astep = 0.1*(alim[1]-alim[0])
	incstep = 0.1*(inclim[1]-inclim[0])
	cstep = 0.1*(clim[1]-clim[0])
	stepsize = np.array([sigmaystep,rpstep,astep,incstep,cstep]) 
	return alphalim, limites, alpha, parametros, alphastep, stepsize
#BIC
def BIC(p):
	nsteps = 10000
	likemax = []
	x = []
	for i in range (1,p):
		alphalim, limites, alpha, parametros, alphastep, stepsize = par_lim(i+1)	
		chain, log_probs, acceptance_rate = metropolis(log_posterior, parametros, alpha, alphalim, limites, stepsize, alphastep, nsteps, i+1)
		likemax.append(np.max(log_probs))
		x.append(i)
	return likemax, x


#xtrae los datos
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
'''
num = []
fig, ax = plt.subplots(ncols=3,nrows=3,sharex=True,figsize=[4*3,2.5*4])
j=k=0
for i in range(0,9):
	plt.sca(ax[j, k])
	ax[j,k].plot(datos[:,0],Z[:,i],lw=2,color='k',label='Componente'+str(i+1))
	num.append(i)
	plt.legend(loc = 'lower left')
	fig.tight_layout()
	k = k+1
    	if (k > 2):
        	k=0
		j=j+1
plt.show()

importantes = eig_vals_sorted/total
plt.plot(num,importantes,lw=2,color='k')
plt.xlabel('$\mathrm{P}$')
plt.ylabel('$\%$')
plt.legend(loc = 'lower left')
plt.show()
'''		
#Numero de componentes principales que se quiere aplicar
#al BIC

p = 9
likemax, x = BIC(p)
bic = []
for i in range(0,len(likemax)):
	bic.append(-2*log(abs(likemax[i]))+(5+x[i])*log(len(datos)))
plt.plot(x,bic,lw=3)
plt.xlabel('$p$')
plt.ylabel('$BIC$')
plt.legend(loc = 'lower left')
plt.show()

p=3
nsteps = 20000
alphalim, limites, alpha, parametros, alphastep, stepsize = par_lim(p)
chain, log_probs, acceptance_rate = metropolis(log_posterior, parametros, alpha, alphalim, limites, stepsize, alphastep, nsteps, p)
promedios = [np.mean(chain[:,0]),np.mean(chain[:,1]),np.mean(chain[:,2]),np.mean(chain[:,3]),np.mean(chain[:,4]),np.mean(chain[:,5]),np.mean(chain[:,6]),np.mean(chain[:,7])]
#constantes = modelo(promedios,t)

