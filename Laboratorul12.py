import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import scipy.stats as stats

with pm.Model() as centered_model:
    a = pm.HalfNormal('a', 10)
    b = pm.Normal('b', 0, a, shape=10)
    idata_cm = pm.sample(2000, target_accept=0.9, return_inferencedata=True, chains=2)

with pm.Model() as non_centered_model:
    a = pm.HalfNormal('a', 10)
    b_offset = pm.Normal('b_offset', mu=0, sd=1, shape=10)
    b = pm.Deterministic('b', 0 + b_offset * a)
    idata_ncm = pm.sample(2000, target_accept=0.9, return_inferencedata=True, chains=2)

# 1. Calculati numărul de lanţuri, mărimea totală a eşantionului generat şi vizualizaţi distribuţia a posteriori.

az.load_arviz_data("centered_eight")
az.load_arviz_data("non_centered_eight")

# numarul de lanturi

chains_centered = idata_cm.posterior.to_array().shape[0]
chains_non_centered = idata_ncm.posterior.to_array().shape[0]

# marimea totala a esantionului generat

sample_size_centered = idata_cm.posterior.to_array().shape[1]
sample_size_non_centered = idata_ncm.posterior.to_array().shape[1]
total_sample_size_centered = chains_centered * sample_size_centered

# vizualizare distributie a posteriori

az.plot_trace(idata_cm, var_names=['a'], divergences='top', compact=False)
az.plot_posterior(idata_cm, var_names=['a'], ref_val=0, rope=[-1, 1], kind='hist', color='C0', credible_interval=0.89, hdi_prob=0.89)

az.plot_trace(idata_ncm, var_names=['a'], divergences='top', compact=False)
az.plot_posterior(idata_ncm, var_names=['a'], ref_val=0, rope=[-1, 1], kind='hist', color='C0', credible_interval=0.89, hdi_prob=0.89)

# 2. Folosiţi ArviZ pentru a compara cele două modele, după criteriile Rˆ (Rhat) şi autocorelaţie. Concentraţi-
# vă pe parametrii mu şi tau 

# rhat

rhat_centered = az.summary(idata_cm, var_names=['a'])
rhat_centered = az.summary(idata_ncm, var_names=['a'])

# autocorelare

autocorrelation_centered = az.autocorrplot(idata_cm, var_names=['b'])
autocorrelation_non_centered = az.autocorrplot(idata_ncm, var_names=['b'])

# 3. Număraţi numărul de divergenţe din fiecare model (cu sample_stats.diverging.sum() ), iar apoi
# identificaţi unde acestea tind să se concentreze în spaţiul parametrilor (mu şi tau ). Puteţi folosi mod-
# elul din curs, cu az.plot pair sau az.plot parallel .

div_nr_centered = idata_cm.sample_stats.diverging.sum()
div_nr_non_centered = idata_ncm.sample_stats.diverging.sum()

az.plot_pair(idata_cm, var_names=['a'], divergences=True, divergences_kwargs={'color':'C1'})
az.plot_pair(idata_ncm, var_names=['a'], divergences=True, divergences_kwargs={'color':'C1'})

