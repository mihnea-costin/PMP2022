import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C', 0.005)
    incendiu = pm.Bernoulli('I', 0.01)
    alarmaincendiu_p = pm.Deterministic('AI_p', pm.math.switch(incendiu, pm.math.switch(cutremur, 0.01, 0.03), pm.math.switch(cutremur, 0.8, 0.2)))
    accidental_p = pm.Deterministic('C_p', pm.math.switch(reducere, pm.math.switch(urgent, 1, 0.5), pm.math.switch(urgent, 0.8, 0.2)))
    neaccidental_p = pm.Deterministic('C_p', pm.math.switch(reducere, pm.math.switch(urgent, 1, 0.5), pm.math.switch(urgent, 0.8, 0.2)))

    cumpara = pm.Bernoulli('C', p=cumpara_p)
    trace = pm.sample(20000)