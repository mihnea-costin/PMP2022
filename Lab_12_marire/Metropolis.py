import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

def post(theta, Y, alpha=1, beta=1):
	if 0 <= theta <= 1:
		prior = stats.beta(alpha, beta).pdf(theta)
		like = stats.bernoulli(theta).pmf(Y).prod()
		prob = like*prior
	else:
		prob =-np.inf
	return prob

Y =stats.bernoulli(0.7).rvs(20)

n_iters = 1000
can_sd_values = [0.2, 1]
alpha = beta = 1
theta = 0.5

for can_sd in can_sd_values:
	trace ={"theta":np.zeros(n_iters)}
	p2 = post(theta, Y, alpha, beta)
	for iter in range(n_iters):
		theta_can = stats.norm(theta, can_sd).rvs(1)
		p1 = post(theta_can, Y, alpha, beta)
		pa =p1/p2
		if pa >stats.uniform(0,1).rvs(1):
			theta = theta_can
			p2 = p1
		trace["theta"][iter] = theta

# a. (3pt.) Folosiţi librăria ArviZ pentru a compara valorile eşantionate folosind diagnostice ca autocorelaţia,
# plot_trace şi ess.

# pentru plot_trace
az.plot_trace(trace["theta"])
plt.show()

# pentru autocorelație
az.plot_autocorr(trace["theta"])
plt.show()

# pentru ess
az.ess(trace["theta"])

# b. (2pt.) Modificaţi codul pentru a obţine mai mult de un lanţ independent. Folosiţi ArviZ pentru a calcula
# statistica rhat.

n_chains = 4

chains = []

for chain in range(n_chains):
    trace = {"theta": np.zeros(n_iters)}
    p2 = post(theta, Y, alpha, beta)
    for i in range(n_iters):
        theta_can = stats.norm(theta, can_sd).rvs(1)
        p1 = post(theta_can, Y, alpha, beta)
        pa = p1/p2
        if pa > stats.uniform(0, 1).rvs(1):
            theta = theta_can
            p2 = p1
        trace["theta"][i] = theta

    chains.append(trace)

# Convert each chain to an InferenceData object
inference_data = [az.from_dict(chain) for chain in chains]

# Merge the chains into a single object
multi_chain = az.concat(*inference_data, dim="chain")

# Calculate Rhat for each parameter
print(az.rhat(multi_chain))