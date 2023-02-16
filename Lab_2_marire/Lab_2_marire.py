# Lab. 1+2: Simularea variabilelor aleatoare. Alegeţi două variabile aleatoare la alegere. Puteţi folosi distribuţii
# standard (de exemplu, normală, exponenţială, Poisson, etc.) sau să le construiţi.

# a. (3pt. - coresp. Lab. 1) Simulaţi 1000 de realizări ale fiecărei variabile aleatoare folosind librăriile
# NumPy sau SciPy din Python.

import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt

# generam 1000 de numere aleatoare cu distributie normala cu media 0 si deviatia standard 1
norm_var = np.random.normal(0, 1, 1000)

# generam 1000 de numere aleatoare cu distributie exponetiala cu parametrul lambda 1
exp_var = np.random.exponential(1, 1000)

print("Primele 10 valori ale variabilei cu distributie normala:")
print(norm_var[:10])

print("Primele 10 valori ale variabilei cu distributie exponențiala:")
print(exp_var[:10])


# b. (3pt.- coresp. Lab. 2) Redaţi cu ajutorul librăriei Matplotlib histogramele variabilelor aleatoare simulate
# şi comparaţi-le cu distribuţiile teoretice.

# Creem distributia teoretica pentru distributia normala
x_norm = np.linspace(-4, 4, 1000)
pdf_norm = norm.pdf(x_norm, 0, 1)

# Creem distributia teoretica pentru distributia exponențiala
x_exp = np.linspace(0, 8, 1000)
pdf_exp = expon.pdf(x_exp, 0, 1)

# Afisam histograma pentru variabila aleatoare normala si distributia teoretica
plt.hist(norm_var, bins=30, density=True, alpha=0.5)
plt.plot(x_norm, pdf_norm, 'r')
plt.title('Distributia normala')
plt.show()

# Afisam histograma pentru variabila aleatoare exponențiala si distributia teoretica
plt.hist(exp_var, bins=30, density=True, alpha=0.5)
plt.plot(x_exp, pdf_exp, 'r')
plt.title('Distributia exponențiala')
plt.show()

# c. (3pt.- coresp. Lab. 2) Determinaţi mediile şi dispersiile variabilelor simulate. Comparaţi-le cu mediile
# şi dispersiile teoretice ale distribuţiilor corespunzătoare.

# Calculam media si dispersia variabilei aleatoare normale simulate
mean_norm = np.mean(norm_var)
var_norm = np.var(norm_var)

# Calculam media si dispersia variabilei aleatoare exponențiale simulate
mean_exp = np.mean(exp_var)
var_exp = np.var(exp_var)

# Calculam media si dispersia teoretice pentru distributia normala
mean_norm_theoretical = 0
var_norm_theoretical = 1

# Calculam media si dispersia teoretice pentru distributia exponențiala
mean_exp_theoretical = 1
var_exp_theoretical = 1

# Afisam media si dispersia variabilelor aleatoare simulate si teoretice pentru distributia normala
print("Variabila aleatoare normala:")
print("Media simulata: ", mean_norm)
print("Media teoretica: ", mean_norm_theoretical)
print("Dispersia simulata: ", var_norm)
print("Dispersia teoretica: ", var_norm_theoretical)

# Afisam media si dispersia variabilelor aleatoare simulate si teoretice pentru distributia exponențiala
print("Variabila aleatoare exponențiala:")
print("Media simulata: ", mean_exp)
print("Media teoretica: ", mean_exp_theoretical)
print("Dispersia simulata: ", var_exp)
print("Dispersia teoretica: ", var_exp_theoretical)

