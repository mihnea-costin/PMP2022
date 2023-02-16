# Lab. 7: Regresie liniară multiplă. Preziceţi valoarea medie a locuinţelor ocupate de proprietari în mii de USD
# în zona Boston pe baza diferiţilor factori, cum ar fi numărul mediu de camere, rata criminalităţii şi suprafaţa
# comercială din oraş. Fişierul BostonHousing.csv conţine un set de date cu observaţii, din care ne interesează
# doar cele corespunzătoare valoarii medii a locuinţelor în mii de USD (medv), numărul mediu de camere (rm),
# rata criminalităţii (crim) şi proporţia suprafaţei comerciale non-retail (indus).

import pandas as pd
from sklearn.linear_model import LinearRegression
import pymc3 as pm
import numpy as np

# a. (1pt.) Încărcaţi setul de date într-un Pandas DataFrame.

data = pd.read_csv("C:\\Users\\mihne\\Desktop\\Facultate\\Anul III\\Programare și modelare probabilistică\\PMP2022\\Lab_7_marire\\BostonHousing.csv", usecols=["medv", "rm", "crim", "indus"])

# b. (3pt.) Definiţi modelul în PyMC3 folosind variabilele independente (rm, crim, indus) pentru a prezice
# variabila dependentă (medv).

# incarcam datele in variabilele X si y
y = data["medv"]
X = data[["rm", "crim", "indus"]]

# definim modelul
with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    betas = pm.Normal("betas", mu=0, sd=10, shape=X.shape[1])
    y_est = alpha + pm.math.dot(X, betas)
    tau = pm.Gamma("tau", alpha=0.1, beta=0.1)
    errors = pm.Normal("errors", mu=0, sd=pm.math.sqrt(1/tau), observed=y - y_est)

# c. (2pt.) Obţineţi estimări de 95% pentru HDI ale parametrilor.

with model:
    trace = pm.sample(5000, tune=1000, random_seed=42)

pm.plot_posterior(trace, var_names=["alpha", "betas", "tau"], credible_interval=0.95)

# d. (3pt.) Simulaţi 5000 de extrageri din distribuţia predictivă posterioară şi utilizaţi aceste extrageri simulate
# pentru a găsi un interval de predicţie de 90% HDI.

with model:
    ppc = pm.sample_posterior_predictive(trace, samples=5000)
    
y_pred = ppc["errors"] + ppc["alpha"] + np.dot(ppc["betas"], X.T)

# calculam intervalul de 90% HDI
y_hdi = pm.stats.hdi(y_pred, hdi_prob=0.9)

# calculam intervalul de 90% HDI
print(y_hdi)

