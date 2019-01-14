import numpy as np
import pandas as pd
import random
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# root mean square error

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# loading the data

df = pd.read_csv("train.csv")
X = df
X = X[X['medv'] < np.quantile(X['medv'], 0.97)]
X = X[X['medv'] > np.quantile(X['medv'], 0.03)]
X = X[X['crim'] < np.quantile(X['crim'], 0.97)]
X = X.reset_index(drop=True)


# dropping unnecessary data

X = X.drop(labels='ID', axis=1)
X = X.drop(labels='chas', axis=1)


# adding quadratic parameters

X['lstat2'] = pd.Series(X['lstat'].pow(2))
X['indus2'] = pd.Series(X['indus'].pow(2))
X['rm2'] = pd.Series(X['rm'].pow(2))
X['age2'] = pd.Series(X['age'].pow(2))
X['black2'] = pd.Series(X['black'].pow(2))


# adding logarithmic parameters

X['lstat_log'] = pd.Series(np.log(X['lstat']))
X['indus_log'] = pd.Series(np.log(X['indus']))
X['rm_log'] = pd.Series(np.log(X['rm']))
X['age_log'] = pd.Series(np.log(X['age']))
X['black_log'] = pd.Series(np.log(X['black']))


# adding column of ones

X['ones'] = pd.Series(np.ones(len(X)))


# separating 'medv' from the rest of the set

Y = X['medv']
X = X.drop(labels='medv', axis=1)


# regularization, as you can see eventually not used

L = np.identity(len(X.columns))
L[0,0] = 0
lamb = 0


# whole magic happens here

theta = np.dot(np.linalg.pinv(X.transpose() @ X + lamb * L) @ X.transpose(), Y)


# preparing the test set

df_test = pd.read_csv('test.csv')
ids_y = df_test['ID']
df_test = df_test.drop(labels='ID', axis=1)
df_test = df_test.drop(labels='chas', axis=1)
df_test['lstat2'] = pd.Series(df_test['lstat'].pow(2))
df_test['indus2'] = pd.Series(df_test['indus'].pow(2))
df_test['rm2'] = pd.Series(df_test['rm'].pow(2))
df_test['age2'] = pd.Series(df_test['age'].pow(2))
df_test['black2'] = pd.Series(df_test['black'].pow(2))
df_test['lstat_log'] = pd.Series(np.log(df_test['lstat']))
df_test['indus_log'] = pd.Series(np.log(df_test['indus']))
df_test['rm_log'] = pd.Series(np.log(df_test['rm']))
df_test['age_log'] = pd.Series(np.log(df_test['age']))
df_test['black_log'] = pd.Series(np.log(df_test['black']))
df_test = df_test.reset_index(drop=True)
df_test['ones'] = pd.Series(np.ones(len(df_test)))


# computing the result

y_test = np.dot(df_test,theta)


# creating a csv file

y_test_df = pd.DataFrame(data=y_test, index=ids_y, columns=['medv'])
y_test_df.to_csv('results2.csv')
