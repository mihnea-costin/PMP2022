import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

# 1. Generaţi 500 de date dintr-o mixtură de trei distribuţii Gaussiene. În fişierul alăturat aveţi un astfel de
# exemplu. (1pt)

# generate 500 data points from a mixture of three Gaussians

'''generate 500 data points from a mixture of three Gaussians'''

import matplotlib.pyplot as plt
import pymc3 as pm

# generate 500 data points from a mixture of three Gaussians

data_mix = np.concatenate([np.random.normal(0, 1, 200), np.random.normal(5, 1, 100), np.random.normal(10, 1, 200)])

# 2. Calibraţi pe acest set de date un model de mixtură de distribuţii Gaussiene cu 2, 3, respectiv 4 compo-
# nente. (3pt)

clusters = 2

with pm.Model() as model_mg:
    p = pm.Dirichlet('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=data_mix.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    idata_mg = pm.sample(random_seed=123, return_inferencedata=True)

# 3. Comparaţi cele 3 modele folosind metodele WAIC şi LOO. Care este concluzia? (3pt)