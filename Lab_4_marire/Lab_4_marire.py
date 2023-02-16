import numpy as np
import pymc3 as pm

# a. (2pt.) Definiţi modelul probabilist (folosind eventual PyMC3 - dar nu obligatoriu, neexistând variabile
# observate) care sa descrie contextul de mai sus.

n = 1000
theta = 0.2
lmbda = 20

def time_spent_in_store(x):
    return (x >= 15)

with pm.Model() as model:
    alpha = pm.Uniform("alpha", lower=0, upper=30)
    n_customers = pm.Poisson("n_customers", mu=lmbda)
    purchase = pm.Binomial("purchase", n=n_customers, p=theta)
    time_no_purchase = pm.Exponential("time_no_purchase", lam=1/alpha, shape=n_customers)
    time_purchase = pm.Exponential("time_purchase", lam=1/(alpha+1), shape=purchase)

    time_spent = pm.math.concatenate([time_no_purchase, time_purchase])
    spent_in_store = pm.Deterministic("spent_in_store", time_spent_in_store(time_spent))

    trace = pm.sample(5000, tune=1000, target_accept=0.9, random_seed=42)

# b. (3pt.) Pentru valori ale lui 𝜃 alese în prealabil (de exemplu, 𝜃 = 0.2 sau 0.5), determinaţi care este (cu
# aproximaţie) 𝛼 maxim pentru ca toţi acei clienţi care nu cumpără să nu stea în magazin mai mult de
# 15 minute, cu o probabilitate de 95%.

time_no_purchase_trace = trace["time_no_purchase"].flatten()
time_no_purchase_trace_95 = np.percentile(time_no_purchase_trace, 95)

alpha_max = 1 / np.percentile(time_no_purchase_trace, 95)
print("α maxim pentru probabilitatea de 95% ca un client care nu cumpără să nu stea mai mult de 15 minute este: ", alpha_max)