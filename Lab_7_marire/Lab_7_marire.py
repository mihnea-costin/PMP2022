# Lab. 7: Regresie liniară multiplă. Preziceţi valoarea medie a locuinţelor ocupate de proprietari în mii de USD
# în zona Boston pe baza diferiţilor factori, cum ar fi numărul mediu de camere, rata criminalităţii şi suprafaţa
# comercială din oraş. Fişierul BostonHousing.csv conţine un set de date cu observaţii, din care ne interesează
# doar cele corespunzătoare valoarii medii a locuinţelor în mii de USD (medv), numărul mediu de camere (rm),
# rata criminalităţii (crim) şi proporţia suprafaţei comerciale non-retail (indus).

# c. (2pt.) Obţineţi estimări de 95% pentru HDI ale parametrilor.
# d. (3pt.) Simulaţi 5000 de extrageri din distribuţia predictivă posterioară şi utilizaţi aceste extrageri simulate
# pentru a găsi un interval de predicţie de 90% HDI.

import pandas as pd
from sklearn.linear_model import LinearRegression
import pymc3 as pm

# a. (1pt.) Încărcaţi setul de date într-un Pandas DataFrame.

data = pd.read_csv("C:\\Users\\mihne\\Desktop\\Costin_Mihnea-Radu_PMP_Marire\\Lab_7_marire\\BostonHousing.csv")

# selectarea variabilelor independente și a variabilei dependente
X = data[['rm', 'crim', 'indus']]
y = data['medv']

# crearea unui model de regresie liniară și antrenarea lui pe datele noastre
model = LinearRegression()
model.fit(X, y)

# prezicerea valorii medii a locuințelor ocupate de proprietari pentru un set de date de test
test_data = pd.DataFrame({'rm': [6.5, 7.2], 'crim': [0.2, 0.5], 'indus': [8, 12]})
predicted_values = model.predict(test_data)

print(predicted_values)

# b. (3pt.) Definiţi modelul în PyMC3 folosind variabilele independente (rm, crim, indus) pentru a prezice
# variabila dependentă (medv).

# definirea modelului în PyMC3
with pm.Model() as model:
    # definirea distribuției normale pentru variabila dependentă
    mu = pm.Normal('mu', mu=0, sigma=10)
    
    # definirea distribuțiilor normale pentru fiecare variabilă independentă și ponderile lor
    beta_rm = pm.Normal('beta_rm', mu=0, sigma=10)
    beta_crim = pm.Normal('beta_crim', mu=0, sigma=10)
    beta_indus = pm.Normal('beta_indus', mu=0, sigma=10)
    
    # definirea modelului liniar
    y_pred = mu + beta_rm * X['rm'] + beta_crim * X['crim'] + beta_indus * X['indus']
    
    # definirea distribuției normale pentru variabila dependentă, dat fiind modelul liniar
    y_obs = pm.Normal('y_obs', mu=y_pred, sigma=1, observed=y)

# c. (2pt.) Obţineţi estimări de 95% pentru HDI ale parametrilor.

# estimari de 95% pentru HDI ale parametrilor

with model:
    # generarea lanțurilor Markov Monte Carlo (MCMC)
    trace = pm.sample(1000, tune=1000)
    
    # afișarea rezumatului lanțurilor MCMC
    print(pm.summary(trace))
    
    # afișarea ploturilor posterioare pentru parametri cu 95 % HDI
    pm.plot_posterior(trace, credible_interval=0.95)


