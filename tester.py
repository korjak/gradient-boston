import numpy as np
import pandas as pd
import random
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def tester(lamb):
    df = pd.read_csv("train.csv")
    X = df
    X = X[X['medv'] < np.quantile(X['medv'], 0.98)]
    X = X[X['medv'] > np.quantile(X['medv'], 0.02)]
    X = X[X['crim'] < np.quantile(X['crim'], 0.98)]
    X = X.reset_index(drop=True)

    cv = pd.DataFrame()
    indices = random.sample(range(len(X)), 60)

    for i in indices:
        cv = cv.append(X.iloc[i])[X.columns.tolist()]

    X = X.reset_index(drop=True)
    X = X.drop(indices)
    X = X.reset_index(drop=True)

    X = X.drop(labels='ID', axis=1)

    cv = cv.drop(labels='ID', axis=1)

    X = X.drop(labels='chas', axis=1)
    cv = cv.drop(labels='chas', axis=1)
    X = X.drop(labels='nox', axis=1)
    cv = cv.drop(labels='nox', axis=1)

    X['lstat2'] = pd.Series(X['lstat'].pow(2))
    cv['lstat2'] = pd.Series(cv['lstat'].pow(2))
    X['indus2'] = pd.Series(X['indus'].pow(2))
    cv['indus2'] = pd.Series(cv['indus'].pow(2))
    X['rm2'] = pd.Series(X['rm'].pow(2))
    cv['rm2'] = pd.Series(cv['rm'].pow(2))
    X['age2'] = pd.Series(X['age'].pow(2))
    cv['age2'] = pd.Series(cv['age'].pow(2))
    X['black2'] = pd.Series(X['black'].pow(2))
    cv['black2'] = pd.Series(cv['black'].pow(2))

    X['lstat_log'] = pd.Series(np.log(X['lstat']))
    cv['lstat_log'] = pd.Series(np.log(cv['lstat']))
    X['indus_log'] = pd.Series(np.log(X['indus']))
    cv['indus_log'] = pd.Series(np.log(cv['indus']))
    X['rm_log'] = pd.Series(np.log(X['rm']))
    cv['rm_log'] = pd.Series(np.log(cv['rm']))
    X['age_log'] = pd.Series(np.log(X['age']))
    cv['age_log'] = pd.Series(np.log(cv['age']))
    X['black_log'] = pd.Series(np.log(X['black']))
    cv['black_log'] = pd.Series(np.log(cv['black']))


    X['ones'] = pd.Series(np.ones(len(X)))
    cv['ones'] = pd.Series(np.ones(len(cv)))

    Y = X['medv']
    X = X.drop(labels='medv', axis=1)

    L = np.identity(len(X.columns))
    L[0, 0] = 0

    theta = np.dot(np.linalg.pinv(X.transpose() @ X + lamb * L) @ X.transpose(), Y)

    cv_y = cv['medv']
    cv = cv.drop(labels='medv', axis=1)
    y_test = np.dot(cv, theta)
    kk = rmse(y_test, cv_y)
    return kk


max_lamb = 0 + 1
mean_rmse_in_iter = 0
mean_rmse = [0] * max_lamb
iter = 100

for lamb in range(max_lamb):
    mean_rmse_in_iter = 0
    for i in range(iter):
        mean_rmse_in_iter += tester(lamb*0.2)
    mean_rmse[lamb] = mean_rmse_in_iter/iter
df_mean_rmse = pd.DataFrame(mean_rmse)
print(df_mean_rmse)