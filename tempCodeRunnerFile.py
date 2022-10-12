import numpy as np
# from scipy import stats

# import statistics
# import matplotlib.pyplot as plt
# import arviz as az
# import random

# np.random.seed(1)

# t1 = stats.gamma(4, 10000, 1/3)
# t2 = stats.gamma(4, 10000, 1/2)
# t3 = stats.gamma(5, 10000, 1/2)
# t4 = stats.gamma(5, 10000, 1/3)

# x = []

# for i in range(1, 10000):
#     y = random.randint(0, 100)
#     l = stats.uniform.rvs(0, 1/4, size=10000)
#     if y < 25:
#         x = t1
#     elif 25 <= y < 50:
#         x = t2
#     elif 50 <= y < 80:
#         x = t3
#     elif y > 80:
#         x = t4

#     x.append(x + l)
# az.plot_posterior({'x':x})

# plt.show() 