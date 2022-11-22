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
        alpha = pm.Normal('a', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=5)
        sigma = pm.HalfNormal('sigma', sd=1)
        # bonus si pentru premium analizat
        mu = alpha + beta1 * speed + beta2 * hardDrive
        price_like = pm.Normal('price_like', mu=mu, sd=sigma, observed=price)
        trace = pm.sample(20000, tune=20000, cores=4)
        ppc = pm.sample_posterior_predictive(trace, samples=100, model=prices_of_pcs)

        az.plot_trace(ppc)
        plt.savefig('ppc.png')
        plt.show()

# 2. Obţineţi estimări de 95% pentru HDI ale parametrilor β1 şi β2. (3pt)
    plt.posterior(trace, var_names=['a', 'beta1', 'beta2', 'gamma', 'sigma'])
    

# 3. Pe baza rezultatelor obţinute, sunt frecvenţa procesorului şi mărimea hard diskului predictori utili ai
# preţului de vânzare?

    # Conform rezultatelor obtinute, frecventa procesorului si marimea hard diskului sunt predictori utili ai 
    # pretului de vanzare deoarece aceste date se reflectă în performanțele produsului, ceea ce conferă un preț mai mare.

# 4.Să presupunem acum că un consumator este interesat de un computer cu o frecvenţă de 33 MHz şi un
# hard disk de 540 MB. Simulaţi 5000 de extrageri din preţul de vânzare aşteptat (μ) şi construiţi un interval de
# 90% HDI pentru acest preţ.



# 5.În schimb, să presupunem că acest consumator doreşte să prezică preţul de vânzare al unui computer cu
# această frecvenţă şi mărime a hard disk-ului. Simulaţi 5000 de extrageri din distribuţia predictivă posterioară
# şi utilizaţi aceste extrageri simulate pentru a găsi un interval de predicţie de 90% HDI.