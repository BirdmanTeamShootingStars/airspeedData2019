from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from math import sqrt

parser = argparse.ArgumentParser()
parser.add_argument("csv_filename")
args = parser.parse_args()

data = pd.read_csv(args.csv_filename)
y = np.array(data['v'].values)
X = np.array([data['p'].values]).T

for i in range(X.size):
    X[i] = sqrt(X[i])

print("X shape: {}" .format(X.shape))
print("y shape: {}" .format(y.shape))
model = LinearRegression().fit(X, y)
print("train score: {}" .format(model.score(X,y)))
print("coef_: {}" .format(model.coef_))
print("intercept_: {}" .format(model.intercept_))

x_domain = np.array([np.linspace(0.15, 0.4)]).T
for i in range(x_domain.size):
    x_domain[i] = sqrt(x_domain[i])

y_predict = model.predict(x_domain)

plt.plot(X, y, 'o')
plt.plot(x_domain, y_predict, 'x')
plt.xlabel('sqrt(p)')
plt.ylabel('v')
plt.show()
