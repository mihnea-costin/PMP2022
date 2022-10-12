#ex.1

import numpy as np
from scipy import stats

import statistics
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

x = stats.binom.rvs(1, 0.6, size=10000) 

m1 = stats.expon.rvs(0, scale= 1/4, size=1)
m2 = stats.expon.rvs(0, scale= 1/6, size=1)

for i in range(10000):
    choose = stats.binom.rvs(n=1, p=0.4, size=1)
    if (x[i]== 1):
        x = np.append(x,m1[0])
    else:
        x = np.append(x,m2[0])

medie = np.mean(x) 
std = np.std(medie)

print(std)
az.plot_posterior({'x':x}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 

#ex.2

import numpy as np
from scipy import stats

import statistics
import matplotlib.pyplot as plt
import arviz as az
import random

np.random.seed(1)

t1 = stats.gamma(4, 10000, 1/8)
t2 = stats.gamma(4, 10000, 1/7)
t3 = stats.gamma(5, 10000, 1/7)
t4 = stats.gamma(5, 10000, 1/8)

x = []

for i in range(1, 10000):
    random = random.randint(0, 100)
    unif = stats.uniform.rvs(0, 1/4, size=10000)
    if random % 25 == 0:
        x.append(x + t1) 
    elif random % 3 == 0:
        x.append(x + t2)
    elif random % 4 == 0:
        x.append(x + t3)
    else:
        x.append(x + t4)

    x.append(x + unif)
    
az.plot_posterior({'x':x})

plt.show() 

#ex3

import numpy as np
from scipy import stats

import statistics
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

x = stats.binom.rvs(1, 0.6, size=10000) 

coin1 = stats.expon.rvs(0, scale= 0.5, size=1)
coin2 = stats.expon.rvs(0, scale= 0.3, size=1)

for i in range(100):
    choose = stats.binom.rvs(n=1, p=0.4, size=1)
    if (x[i]== 1):
        x = np.append(x,coin1[0])
    else:
        x = np.append(x,coin2[0])

medie = np.mean(x) 
std = np.std(medie)

print(std)
az.plot_posterior({'Moneda 1':coin1,'Moneda 2':coin2}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 

