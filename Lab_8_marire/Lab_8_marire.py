import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# a. (2pt.) Încărcaţi setul de date Titanic.csv într-un Pandas DataFrame şi preprocesaţi datele prin gestionarea
# valorilor lipsă, transformarea variabilelor (dacă este necesar), etc.

# incarcam setul de date Titanic.csv in Pandas DataFrame
df = pd.read_csv("\\Users\\mihne\\Desktop\\Facultate\\Anul III\\Programare și modelare probabilistică\\PMP2022\\Lab_8_marire\\Titanic.csv")
print(df.head())

# identificam valorile lipsa
print(df.isnull().sum())

df.drop("Cabin", axis=1, inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)

# verificam daca trebuie sa transformam variabilele in numerice
print(df.dtypes)

# variabilele sunt de tip string, deci trebuie sa le transformam in numerice
df["Sex"] = df["Sex"].replace({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].replace({"S": 0, "C": 1, "Q": 2})
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# extragem datele relevante pentru model
X = df[["Pclass", "Age"]].values
y = df["Survived"].values

# imapartim setul de date in date de antrenare si date de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# antrenam modelul de clasificare pe datele de antrenare e
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# extragem datele relevante pentru model
X = df[["Pclass", "Age"]].values
y = df["Survived"].values

# b. (2pt.) Definiţi modelul în PyMC3 folosind cele două variabile independente (clasa de pasageri: Pclass
# şi vârsta: Age) pentru a prezice variabila dependentă (Survived).

# definirea modelului logistic în PyMC3
with pm.Model() as model:
    beta_0 = pm.Normal("beta_0", mu=0, sigma=10)
    beta_1 = pm.Normal("beta_1", mu=0, sigma=10)
    beta_2 = pm.Normal("beta_2", mu=0, sigma=10)
    z = beta_0 + beta_1*X[:,0] + beta_2*X[:,1]
    p = pm.Deterministic("p", 1 / (1 + np.exp(-z)))
    observed = pm.Bernoulli("observed", p=p, observed=y)
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step)

# vizualizarea rezultatelor
pm.traceplot(trace)

# c. (2pt.) Care este, în medie, graniţa de decizie pentru acest model? Reprezentaţi de asemenea grafic o
# zonă în jurul acestei grafic care să reprezinte un interval 95% HDI.

with model:
    trace = pm.sample(draws=2000, tune=1000, chains=2)
    
    # vizualizarea rezultatelor
    az.plot_posterior(trace, hdi_prob=0.95)

# calcularea granitei de decizie
intercept = np.mean(trace['Intercept'])
pclass_mean = np.mean(trace['Pclass'])
age_mean = np.mean(trace['Age'])
boundary_age = -intercept / age_mean / 2
print('Decision boundary age:', boundary_age)

# d. (2pt.) Care credeţi că este variabila care influenţează cel mai mult rezultatul (dacă pasagerul a supravieţuit
# sau nu)?

# Bazându-ne pe modelul construit cu ajutorul PyMC3, putem concluziona că clasa de pasageri este cel mai important factor care determină șansele de 
# supraviețuire în cazul naufragiului de pe Titanic, în comparație cu vârsta pasagerilor. Cu alte cuvinte, șansele de supraviețuire erau mai mari 
# pentru pasagerii care călătoreau în clase superioare (clasa 1), decât pentru cei care călătoreau în clase inferioare (clasele 2 și 3). 
# Deși vârsta a avut și ea o influență asupra șanselor de supraviețuire, aceasta nu a fost atât de mare precum clasa de pasageri.



