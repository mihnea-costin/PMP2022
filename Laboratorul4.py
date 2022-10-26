import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

# 1) modelul probabilist

nr_clienti = stats.Poisson.rvs(0,20, size=10000) # Distributie Poisson a numarului de clienti care pot intra in restaurant
timp_plasare_plata = stats.normal.rvs(0,1,1/2)
pregatire = stats.expon.rvs(0, 1/20 , size=10000)

# 2) Determinaţi care este (cu aproximaţie) α maxim pentru a le putea servi mancarea intr-un timp mai
# scurt de 15 minute tuturor clienţilor care intră într-o oră, cu o probabilitate de 95%. 

alpha_maxim_pentru_servire_rapida = stats.norm.ppf(0.95, 0, 1/2)

# 3) Cu α astfel calculat, determinaţi timpul mediu de aşteptare pentru a fi servit al unui client.

timpul_mediu_de_asteptare= stats.norm.interval(alpha_maxim_pentru_servire_rapida, 0, 1/2)

# 4) Bonus: Generaţi un eşantion de 100 de timpi de aşteptare medii dintre care 95% sunt sub 15 minute şi
# cu aceste date inferaţi asupra lui α. Este acest α comparabil cu cel găsit mai sus?
