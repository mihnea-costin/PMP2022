import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

# 1) modelul probabilist

nr_clienti = stats.Poisson.rvs(0,20, size=10000) # Distributie Poisson a numarului de clienti care pot intra in restaurant
timp_plasare_plata = stats.normal.rvs(0,1,1/2)
pregatire = stats.expon.rvs(0, 1/20 , size=10000)

# 2)

# 3)

# 4)
