#ex.1

# import numpy as np
# from scipy import stats

# import statistics
# import matplotlib.pyplot as plt
# import arviz as az

# np.random.seed(1)

# x = stats.binom.rvs(1, 0.6, size=10000) # Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
# a = []

# for i in range(0, 10000):
#     choose = stats.binom.rvs(n=1, p=0.4, size=1)
#     if choose == 1:
#         # x.append(stats.expon.rvs(0, scale= 1/4, size=1)[0])
#         x = np.append(x,stats.expon.rvs(0, scale= 1/4, size=1)[0])
#     else:
#         x = np.append(x,stats.expon.rvs(0, scale= 1/6, size=1)[0])

# medie = np.mean(x) 
# std = np.std(medie)

# print(std)
# az.plot_posterior({'x':x}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
# plt.show() 

#ex.2

# distributia gama(alpha,lambda) se poate apela cu stats.gamma(alpha,0,1/lambda)
# sau stats.gamma(alpha, scale=1/lambda)

#ex3

import numpy as np
from scipy import stats

import statistics
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(10)

x = stats.binom.rvs(1, 0.6, size=10000)
a = []

for i in range(0, 100):
    choose = stats.binom.rvs(n=1, p=0.3, size=2)
    if choose == 1:
        # x.append(stats.expon.r vs(0, scale= 1/4, size=1)[0])
        x = np.append(x,stats.expon.rvs(0, scale= 1/5, size=1)[0])
    else:
        x = np.append(x,stats.expon.rvs(0, scale= 1/3, size=1)[0])

medie = np.mean(x) 
std = np.std(medie)

print(std)
az.plot_posterior({'x':x}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 

