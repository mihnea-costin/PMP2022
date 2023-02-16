# Lab. 4: Modele probabiliste. Presupunem că numărul 𝑛 de clienţi care intră într-o anumită oră într-un
# magazin umează o distribuţie Poisson de parametru 𝜆 = 20 clienţi. Numărul 𝑌 de clienţi care fac cumpărături
# e distribuit Binomial(𝑛, 𝜃), unde 𝜃 este probabilitatea ca un client să cumpere din magazin.
# Să presupunem că un client petrece în magazin un timp distribuit exponenţial cu medie de 𝛼 minute dacă
# nu face cumpărături, respectiv 𝛼 + 1 minute dacă face cumpărături.


# c. (2pt.) Pentru 𝛼 găsit mai sus, care este timpul total petrecut în magazin de toţi clienţii?
import pymc3 as pm
import numpy as np

# a. (2pt.) Definiţi modelul probabilist (folosind eventual PyMC3 - dar nu obligatoriu, neexistând variabile
# observate) care sa descrie contextul de mai sus.

# Valoarea lui theta aleasa in prealabil
theta = 0.5

# Definim modelul probabilistic
with pm.Model() as model:
    # Variabila aleatoare pentru numarul de clienti intr-o ora
    n = pm.Poisson('n', mu=20)
    
    # Variabila aleatoare pentru probabilitatea de a face cumparaturi
    theta = pm.Beta('theta', alpha=1, beta=1, testval=theta)
    
    # Variabila aleatoare pentru numarul de clienti care fac cumparaturi
    y = pm.Binomial('y', n=n, p=theta)
    
    # Variabile aleatoare pentru timpul petrecut in magazin
    alpha = pm.Exponential('alpha', lam=1)
    beta = pm.Exponential('beta', lam=1+1)
    
    # Functie pentru calculul timpului total petrecut in magazin
    @pm.deterministic
    def total_time(y=y, alpha=alpha, beta=beta):
        return (n - y) * alpha + y * beta
    
    # Restrictia de timp pentru clientii care nu cumpara
    max_time = 15
    
    # Determinam alfa maxim astfel incat toti clientii care nu cumpara sa nu stea in magazin mai mult de 15 minute, cu o probabilitate de 95%
    alpha_max = pm.find_MAP(vars=[alpha], fmin=pm.find_MAP, 
                            start={'alpha': 1}, 
                            f=lambda x: np.abs(pm.Exponential.dist(lam=x).ppf(0.95) - max_time),
                            progressbar=False)['alpha']
    
    # Generam 1000 de esantioane din distributia noastra
    trace = pm.sample(1000, tune=1000)
    
# Extragere cuantile din distributia timpului total petrecut in magazin
q = pm.quantiles(trace['total_time'], q=[0.95])

# Verificare conditie
if q[0.95] <= max_time:
    print(f"Valoarea maxima pentru alpha este {alpha_max:.2f} minute")
else:
    print("Nu exista o valoare maxima pentru alpha astfel incat toti clientii care nu cumpara sa nu stea in magazin mai mult de 15 minute, cu o probabilitate de 95%")

