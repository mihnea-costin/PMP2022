import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import seaborn as sns
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az


# 1. Folosind distribuţii a priori slab informative asupra parametrilor β0, β1 şi β2, folosiţi PyMC3 pentru a
# simula un eşantion suficient de mare (construi modelul) din distribuţia a posteriori.

with pm.Model() as model1:
    alpha = pm.Normal('α', mu=0, sd=5)
    beta = pm.Normal('β', mu=0, sd=5)
    μ = alpha + pm.math.dot(beta), β)
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
    bd = pm.Deterministic('bd', -alpha/beta)
    yl = pm.Bernoulli('yl', p=θ, observed=y_0)
idata_0 = pm.sample(1000, return_inferencedata=True)

# 2. Care este, în medie, graniţa de decizie pentru acest model? Reprezentaţi de asemenea grafic o zonă în
# jurul acestei grafic care să reprezinte un interval 94% HDI.

alpha_min, beta_min = alpha.min(), alpha.max()
alpha_grid = np.linspace(alpha_min, beta_min, n)
with pm.Model() as model2:
    ppc = pm.sample_posterior_predictive(idata_0.posterior, samples=1000)
bd = ppc['bd']
bd_mean = bd.mean(axis=0)
bd_hdi = az.hdi(bd, hdi_prob=0.94)
plt.plot(alpha_grid, bd_mean, color='k', label='mean')
plt.fill_between(alpha_grid, bd_hdi[:, 0], bd_hdi[:, 1], color='k', alpha=0.2, label='94% HDI')
plt.scatter(alpha, beta, c=beta, cmap='bwr', alpha=0.95)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.legend()
plt.show()

# 3. Să presupunem că un student are un scor GRE de 550 şi un GPA de 3.5. Construiţi un interval de 90%
# HDI pentru probabilitatea ca acest student să fie admis.

gre = 550
gpa = 3.5
student_admission_prb_1 = logistic(alpha + beta[0] * gre + beta[1] * gpa)
stud_admission_prb_hdi_1 = az.hdi(student_admission_prb, hdi_prob=0.90)
print(stud_admission_prb_hdi_1)

# 4. 

# 4. Dar dacă studentul are un scor GRE de 500 şi un GPA de 3.2? (refaceţi exerciţiul anterior cu aceste date)
# Cum justificaţi diferenţa?

gre = 500
gpa = 3.2
student_admission_prb_2 = logistic(alpha + beta[0] * gre + beta[1] * gpa)
stud_admission_prb_hdi_2 = az.hdi(student_admission_prb, hdi_prob=0.90)
print(stud_admission_prb_hdi_2)

# Aceasta diferenta se justifica datorita studentul are probabilitate mai mare de admitere daca are scor GRE mai mare si GPA mai mare
# deci din acest motiv rezultatul este mai mare in primul caz.