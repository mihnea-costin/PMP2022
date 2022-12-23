import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. (1pt) Folosiţi metoda grid computing cu alte distribuţii a priori ca cea din curs.
def posterior_grid(grid_points=50, heads=6, tails=9):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) # uniform prior
    prior = (grid<= 0.5).astype(int)
    # prior = abs(grid - 0.5)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

print(posterior_grid())

# 2. (2pt) În codul folosit pentru estimarea lui π, fixaţi-l pe N şi rulaţi codul de mai multe ori. Veţi observa
# că rezultatele sunt diferite, deoarece folosim numere aleatoare.
# Puteţi estima care este legătura între numărul N de puncte şi eroare? Pentru o mai bună estimare,
# va trebui să modificaţi codul pentru a calcula eroarea ca funcţie de N.
# Puteţi astfel rula codul de mai multe ori cu acelaşi N (încercaţi, de exemplu N = 100, 1000 şi 10000),
# calcula media şi deviaţia standard a erorii, iar rezultatele le puteţi vizualiza cu funcţia plt.errorbar()
# din matplotlib

def function(N):
    # N = 10000
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    outside = np.invert(inside)
    plt.figure(figsize=(8, 8))
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label=f'π*= {pi:4.3f}\n error = {error:4.3f}', alpha=0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc=1, frameon=True, framealpha=0.9)
    return error

# print(function(10000))

# 3. (2pt) Modificaţi argumentul func din funcţia metropolis din curs folosind parametrii distribuţiei a
# priori din Cursul 2 (pentru modelul beta-binomial) şi comparaţi cu metoda grid computing.

def metropolis(func, draws=10000):
    """A very simple Metropolis implementation"""
    trace = np.zeros(draws)
    old_x = 0.5 # func.mean()
    old_prob = func.pdf(old_x)
    delta = np.random.normal(0, 0.5, draws)
    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = func.pdf(new_x)
        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x
    return trace