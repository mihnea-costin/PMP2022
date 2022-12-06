import random
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

# (1) numarul minim de case de pus comanda si numarul minim de statii de gatit necesare pentru ca localul
# sa poata servi mancarea tuturor clientilor, intr-un timp mai scurt de 15 minute, cu o probabilitate de
# 95% (3pt);

nr_minim_case = stats.norm.ppf(0.95, 0, 1/2)
print(nr_minim_case)

# (2) numarul minim de mese necesare pentru ca toti clientii care intra in local sa poata sa stea la masa sa si sa isi manance mancarea cu o probabilitate de 90% (2pt);

nr_minim_mese = stats.norm.ppf(0.9, 0, 1/2)
print(nr_minim_mese)
