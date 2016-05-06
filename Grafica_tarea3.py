import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpltools import style
from mpltools import layout


x = np.linspace(-9, 9)
plt.plot(x, st.norm.pdf(x, 0, np.sqrt(1)), label = 'PDF de $\hat{\\theta}$', color = 'b', lw = float (2.0))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.axvline(x=0, lw = float(1.0), ls = 'dotted', color = 'r', label = '$\mathbb{E}(\hat{\\theta})$')
plt.text(0.1, 0.00, r'$\mathbb{E}(\hat{ \theta})$', fontsize=15.0)
plt.ylim((-0.01,0.42))
plt.axvline(x=-1.5, lw = float(1.0), ls = 'dashed', color ='c', label = 'Valor de $\\theta$')
plt.legend(loc = 2)
plt.text(-1.4, 0.00, r'$\theta$', fontsize=15.0)
plt.savefig('normal.eps')
#plt.show()
plt.close()
x = np.linspace(-9, 9)
plt.plot(x, st.norm.pdf(x, 0, np.sqrt(1)), label = 'PDF de $\hat{\\theta}$', color = 'b', lw = float (2.0))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off
plt.axvline(x=0, lw = float(1.0), ls = 'dotted', color = 'r', label = '$\mathbb{E}(\hat{\\theta})$')
plt.text(0.2, 0.00, r'$\mathbb{E}(\hat{ \theta})$', fontsize=15.0)
plt.ylim((-0.01,0.42))
plt.axvline(x=0, lw = float(1.0), ls = 'dashed', color ='c', label = 'Valor de $\\theta$')
plt.legend(loc = 2)
plt.text(-0.7, 0.00, r'$\theta$', fontsize=15.0)
plt.savefig('normal_sesgo.eps')
