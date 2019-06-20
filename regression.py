from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv_filename")
args = parser.parse_args()

data = pd.read_csv(args.csv_filename)
X = np.array([data['v'].values]).T
y = np.array(data['p'].values)

for i in range(y.size):
    y[i] = y[i]**2

print("X shape: {}" .format(X.shape))
print("y shape: {}" .format(y.shape))
model = LinearRegression().fit(X, y)
print("train score: {}" .format(model.score(X,y)))
print("coef_: {}" .format(model.coef_))
print("intercept_: {}" .format(model.intercept_))
