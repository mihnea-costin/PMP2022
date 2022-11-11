import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv("C:\\Users\\mihne\\Desktop\\PMP2022\\Prices.csv")

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values
    
    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0,0].scatter(speed, price, alpha=0.6)
    axes[0,1].scatter(hardDrive, price, alpha=0.6)
    axes[1,0].scatter(ram, price, alpha=0.6)
    axes[1,1].scatter(premium, price, alpha=0.6) 
    axes[0,0].set_ylabel("Price")
    axes[0,0].set_xlabel("Speed")
    axes[0,1].set_xlabel("HardDrive")
    axes[1,0].set_xlabel("Ram")
    axes[1,1].set_xlabel("Premium")
    # plt.savefig('price_correlations.png')
    # plt.show()


# 1. Folosind distribuţii a priori slab informative asupra parametrilor α, β1, β2 şi σ, folosiţi PyMC3 pentru a
# simula un eşantion suficient de mare (construi modelul) din distribuţia a posteriori. (2pt)
    
    prices_of_pcs = pm.Model()

    with prices_of_pcs:
        alpha = pm.Normal('a', mu=0, sd=100)
        beta1 = pm.Normal('beta1', mu=0, sd=100)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        gamma = pm.Normal('gamma', mu=0, sd=100)
        sigma = pm.HalfNormal('sigma', sd=10)
        
        mu = alpha + beta1 * speed + beta2 * hardDrive + gamma * premium
        price_like = pm.Normal('price_like', mu=mu, sd=sigma, observed=price)
        trace = pm.sample(20000, tune=20000, cores=4)
        ppc = pm.sample_posterior_predictive(trace, samples=100, model=prices_of_pcs)

        az.plot_trace(trace)
        plt.savefig('trace.png')
        plt.show()

# 2. Obţineţi estimări de 95% pentru HDI ale parametrilor β1 şi β2. (3pt)
    # plt.posterior(trace, var_names=['a', 'beta1', 'beta2', 'gamma', 'sigma'])

