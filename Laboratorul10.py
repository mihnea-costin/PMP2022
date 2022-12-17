import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

# 1. Generaţi 500 de date dintr-o mixtură de trei distribuţii Gaussiene. În fişierul alăturat aveţi un astfel de
# exemplu. (1pt)

data_mix = np.concatenate([np.random.normal(0, 1, 200), np.random.normal(5, 1, 100), np.random.normal(10, 1, 200)])

# 2. Calibraţi pe acest set de date un model de mixtură de distribuţii Gaussiene cu 2, 3, respectiv 4 compo-
# nente. (3pt)

clusters_1 = 2

with pm.Model() as model_mix_1:
    d1 = pm.Normal('d1', mu=0, sd=1)
    d2 = pm.Normal('d2', mu=5, sd=1)
    d3 = pm.Normal('d3', mu=10, sd=1)
    m = pm.NormalMixture('m', w=np.array([1 / clusters_1] * clusters_1), mu=[d1, d2], sd=[1, 1], observed=data_mix)
    idata_mix_1 = pm.sample(random_seed=123, return_inferencedata=True)

clusters_2 = 3

with pm.Model() as model_mix_2:
    d1 = pm.Normal('d1', mu=0, sd=1)
    d2 = pm.Normal('d2', mu=5, sd=1)
    d3 = pm.Normal('d3', mu=10, sd=1)
    m = pm.NormalMixture('m', w=np.array([1 / clusters_2] * clusters_2), mu=[d1, d2, d3], sd=[1, 1, 1], observed=data_mix)
    idata_mix_2 = pm.sample(random_seed=123, return_inferencedata=True)

clusters_3 = 4

with pm.Model() as model_mix_3:
    d1 = pm.Normal('d1', mu=0, sd=1)
    d2 = pm.Normal('d2', mu=5, sd=1)
    d3 = pm.Normal('d3', mu=10, sd=1)
    m = pm.NormalMixture('m', w=np.array([1 / clusters_3] * clusters_3), mu=[d1, d2, d3, 15], sd=[1, 1, 1, 1], observed=data_mix)
    idata_mix_3 = pm.sample(random_seed=123, return_inferencedata=True)

# 3. Comparaţi cele 3 modele folosind metodele WAIC şi LOO. Care este concluzia? (3pt)

# WAIC

waic_1 = az.waic(idata_mix_1, model=model_mix_1, scale='deviance')
waic_2 = az.waic(idata_mix_2, model=model_mix_2, scale='deviance')
waic_3 = az.waic(idata_mix_3, model=model_mix_3, scale='deviance')

cmp_waic = az.compare({'model_mix_1': idata_mix_1, 'model_mix_2': idata_mix_2, 'model_mix_3': idata_mix_3}, method='BB-pseudo-BMA', ic = 'waic', scale='deviance')

# LOO

loo_1 = az.loo(idata_mix_1, model=model_mix_1, scale='deviance')
loo_2 = az.loo(idata_mix_2, model=model_mix_2, scale='deviance')
loo_3 = az.loo(idata_mix_3, model=model_mix_3, scale='deviance')

cmp_loo = az.compare({'model_mix_1': idata_mix_1, 'model_mix_2': idata_mix_2, 'model_mix_3': idata_mix_3}, method='BB-pseudo-BMA', ic = 'loo', scale='deviance')

# Concluzia este ca cel mai bun model este cel cu 3 clustere, deoarece are scorurile pentru WAIC si LOO cele mai mici.