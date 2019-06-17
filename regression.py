from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

data = pd.read_csv('./20190509.csv')
X = np.array([data['v'].values]).T
y = np.array(data['p'].values)

for i in range(y.size):
    y[i] = y[i]**2

print("X shape: {}" .format(X.shape))
print("y shape: {}" .format(y.shape))
ridge = Ridge(alpha=0.0).fit(X, y)
print("train score: {}" .format(ridge.score(X,y)))
print("coef_: {}" .format(ridge.coef_))
