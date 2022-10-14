import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

#1) modelul probabilist
with model:
    cutremur = pm.Bernoulli('C', 0.0005)

    incendiu_p = pm.Deterministic('I_p', pm.math.switch(cutremur, 0.01, 0.03))
    incendiu = pm.Bernoulli('I', incendiu_p)

    alarma_p = pm.Deterministic('A_p', pm.math.switch(incendiu, pm.math.switch(cutremur, 0.01, 0.02), pm.math.switch(cutremur, 0.95, 0.98)))
    alarma = pm.Bernoulli('C', p=alarma_p, observed=1)

    trace = pm.sample(20000)

dictionary = {
          'cutremur': trace['C'].tolist(),
          'incendiu': trace['I'].tolist(),
          'alarma': trace['A'].tolist()
          }

df = pd.DataFrame(dictionary)    

#2) prob cutremur 

p_cutremur = df[(df['cutremur'] == 1)].shape[0] / df.shape[0]
print(p_cutremur)

#3) prob incendiu

p_incendiu = df[(df['incendiu'] == 1)].shape[0] / df.shape[0]
print(p_incendiu)

#4) calculati cele 2 probabilitati folosind regula lui Bayes

