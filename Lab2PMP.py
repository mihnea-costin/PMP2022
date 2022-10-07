#ex.1

import numpy as np
from scipy import stats

import statistics
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

lambda1=4
lambda2=6

x = stats.uniform.rvs(-1, 2, size=10000) # Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru fiind limita inferioara a intervalului, al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1] 
#y=(lambda1+1,4*lambda2)/2 # Compunerea prin insumare a celor 2 distributii

m1=stats.expon(0,1/lambda1)
m2=stats.expon(0,1/lambda2)

x=1,5*lambda1+lambda2

print("Deviatia standard a lui x este % s "% (statistics.stdev(x)))
print("Media lui x este % s " % (statistics.mean(x))) 

#y=(m1*1,4*m2)/2

az.plot_posterior({'x':x,'m1':m1,'m2':m2}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show() 


