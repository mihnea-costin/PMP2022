# 1. Reprezentaţi grafic datele care dau dependenţa rezultatului testului de vârsta mamei (1pt - deadline: sfârşi-
# tul seminarului)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd
import pymc3 as pm

# Preluare date din fisierul csv
df = pd.read_csv("C:\\Users\\mihne\\Desktop\\PMP2022\\data.csv")

Y = df['momage']
X = df['ppvt']

X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)

# Plot 
plt.scatter(X, Y,  color='black')
plt.title('Test Data')
plt.xlabel('Mother Age')
plt.ylabel('IQ')
plt.xticks(())
plt.yticks(())

#Afisare grafic
plt.show()

# 2. Definiţi modelul Bayesian de regresie liniară (folosind PyMC3) care sa descrie contextul de mai sus. (2pt -
# deadline: sfârşitul seminarului)

with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=50)
    β = pm.Normal('β', mu=0, sd=10)
    ε = pm.HalfCauchy('ε', 5)
    μ = pm.Deterministic('μ', α + β * X)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=Y)
    
    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)


# Observatie: nu este nevoie sa folosim pm.Deterministic decat dac ̆a vrem ca variabila
# μ sa fie calculat ̆a  ̧si salvat ̆a ˆın e ̧santion
# .
# 3. Determinaţi care este dreapta de regresie care se potriveşte cel mai bine datelor. La ce vârstă aţi recomanda
# mamelor să nască? (2pt - deadline: sfârşitul seminarului)

posterior_g = idata_g.posterior.stack(samples={"educ_cat", "ppvt"})

# Afisare dreapta de regresie
print(posterior_g)

# La ce varsta as recomanda mamelor sa nasca?

# Eu as recomanda mamelor sa nasca la o varsta de minim 25 de ani deoarece nivelul educatiei este mai mare si se poate face o mai buna pregatire a organismului pentru sarcina.

# az.plotspa
# 4. Repetaţi acum cerinţele de mai sus schimbând “vârsta” cu “nivelul de educaţie al” mamei. S-au schimbat
# concluziile relativ la momentul naşterii? (3pt - deadline: luni, 7.11.2022)

Y = df['educ_cat']
X = df['ppvt']

X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)

# Plot 
plt.scatter(X, Y,  color='black')
plt.title('Test Data')
plt.xlabel('Education Level')
plt.ylabel('IQ')
plt.xticks(())
plt.yticks(())

#Afisare grafic
plt.show()

#Modelul Bayesian de regresie liniara

with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=50)
    β = pm.Normal('β', mu=0, sd=10)
    ε = pm.HalfCauchy('ε', 5)
    μ = pm.Deterministic('μ', α + β * X)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=Y)
    
    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)

# Afisare dreapta de regresie
print(posterior_g)

# Bonus: Reprezentaţi grafic un set de 100 de date generate conform distribuţiei predictive a posteriori, îm-
# preună cu regiunea 97%HPD pentru acestea (1pt pentru fiecare caz, cu deadline-urile respective: sfârşitul
# seminarului şi luni, 7.11.2022).

