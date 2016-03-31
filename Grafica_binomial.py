import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

rv = st.binom(33,0.5)
k = np.arange(34)
pk = rv.pmf(k) #pmf: probability mass function
rv1 = st.norm(16.5,8.25)

plt.vlines(k, 0, pk)
plt.plot(k, pk, 'o', label = '$\sim$Binomial(33,0.5)')
plt.xlabel('$x$')
plt.ylabel('$f_X(x)$')
#x = np.linspace(-3.0, 35.0)
#plt.plot(x, st.norm.pdf(x, 16.5, np.sqrt(8.25)), label = '$\sim$Normal(16.5,8.25)')
plt.legend(loc = 2)
plt.show()

