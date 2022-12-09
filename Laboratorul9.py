import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

# 1.
# Pe modelul polinomial din curs, în codul care generează datele (din fişierul date.csv ), schimbaţi
# order=2 cu o altă valoare, de exemplu order=5.

data3 = np.loadtxt("C:\\Users\\mihne\\Desktop\\PMP2022\\date.csv")
x_1 = data3[:, 0]
y_1 = data3[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# a. Faceţi apoi inferenţa cu model_p şi reprezentaţi grafic această curbă.

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig('C:\\Users\\mihne\\Desktop\\PMP2022\\model_p.png')

# Repetaţi, dar folosind o distribuţie pentru beta cu sd=100 în loc de sd=10 . În ce fel sunt curbele
# diferite? Încercaţi acest lucru şi cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1]) .

with pm.Model() as model_p1:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p1 = pm.sample(2000, return_inferencedata=True)

α_p1_post = idata_p1.posterior['α'].mean(("chain", "draw")).values
β_p1_post = idata_p1.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p1_post = α_p1_post + np.dot(β_p1_post, x_1s)

plt.plot(x_1s[0][idx], y_p1_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig('C:\\Users\\mihne\\Desktop\\PMP2022\\model_p2.png')

# Curbele sunt diferite in sensul in care cea din primul model are o tendinta de a se apropia de 
# de punctele reprezentate decat in cel de-al doilea model.

with pm.Model() as model_p2:
    α = pm.Normal('α', mu=0, sd = np.array([10, 0.1, 0.1, 0.1, 0.1]))
    β = pm.Normal('β', mu=0, sd=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p2 = pm.sample(2000, return_inferencedata=True)

α_p2_post = idata_p2.posterior['α'].mean(("chain", "draw")).values
β_p2_post = idata_p2.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p2_post = α_p2_post + np.dot(β_p2_post, x_1s)

plt.plot(x_1s[0][idx], y_p2_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig('C:\\Users\\mihne\\Desktop\\PMP2022\\model_p3.png')

# 2. Repetaţi exerciţiul precedent, dar creşteţi numărul de date la 500 de puncte. (2pt)
# Pe modelul polinomial din curs, în codul care generează datele (din fişierul date.csv ), schimbaţi
# order=2 cu o altă valoare, de exemplu order=5.

data3 = np.loadtxt("C:\\Users\\mihne\\Desktop\\PMP2022\\date2.csv")
x_1 = data3[:, 0]
y_1 = data3[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# a. Faceţi apoi inferenţa cu model_p şi reprezentaţi grafic această curbă.

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig('C:\\Users\\mihne\\Desktop\\PMP2022\\model_p.png')

# Repetaţi, dar folosind o distribuţie pentru beta cu sd=100 în loc de sd=10 . În ce fel sunt curbele
# diferite? Încercaţi acest lucru şi cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1]) .

with pm.Model() as model_p1:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p1 = pm.sample(2000, return_inferencedata=True)

α_p1_post = idata_p1.posterior['α'].mean(("chain", "draw")).values
β_p1_post = idata_p1.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p1_post = α_p1_post + np.dot(β_p1_post, x_1s)

plt.plot(x_1s[0][idx], y_p1_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig('C:\\Users\\mihne\\Desktop\\PMP2022\\model_p2.png')

# Curbele sunt diferite in sensul in care cea din primul model are o tendinta de a se apropia de 
# de punctele reprezentate decat in cel de-al doilea model.

with pm.Model() as model_p2:
    α = pm.Normal('α', mu=0, sd = np.array([10, 0.1, 0.1, 0.1, 0.1]))
    β = pm.Normal('β', mu=0, sd=100, shape=order)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p2 = pm.sample(2000, return_inferencedata=True)

α_p2_post = idata_p2.posterior['α'].mean(("chain", "draw")).values
β_p2_post = idata_p2.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p2_post = α_p2_post + np.dot(β_p2_post, x_1s)

plt.plot(x_1s[0][idx], y_p2_post[idx], 'C2', label=f'model order {order}')

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig('C:\\Users\\mihne\\Desktop\\PMP2022\\model_p3.png')


# 3. Faceţi inferenţa cu un model cubic (order=3 ), calculaţi WAIC şi LOO, reprezentaţi grafic rezultatele
# şi comparaţi-le cu modelele liniare şi pătratice. (3pt)

data3 = np.loadtxt("C:\\Users\\mihne\\Desktop\\PMP2022\\date2.csv")
x_1 = data3[:, 0]
y_1 = data3[:, 1]
order = 5
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

with pm.Model() as model_l:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10)
    ε = pm.HalfNormal('ε', 5)
    μ = α + β * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

# calculam WAIC si LOO
waic_l = az.waic(idata_l, scale="deviance")
loo_l = az.loo_pit(idata_l, scale="deviance")
az.show_models(idata_l, scale="deviance")

# reprezentam grafic rezultatele si comparam cu modelele liniare si patratice
cmp_df = az.compare({'model_l':idata_l, 'model_p':idata_p},method='BB-pseudo-BMA', ic="waic", scale="deviance")
az.plot_compare(cmp_df, insample_dev=False, plot_ic_diff=True, figsize=(35, 1))
