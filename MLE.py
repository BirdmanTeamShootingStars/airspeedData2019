import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *

def predict(alpha, beta, p):
    return beta*p + alpha

data = pd.read_csv('20190509.csv')

v = data['v'].values
p = data['p'].values

double_v = v**2

plt.scatter(double_v, p)
plt.show()

print(predict(10,1,p))
#Gauss

