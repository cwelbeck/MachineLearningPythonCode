
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)
plt.rc('font', size=14)
import statsmodels.formula.api as smf

PortBank = pd.read_csv("PortBank.csv")
numerical   ::: SelectColumns = PortBankL[['age', 'duration', 'emp_var_rate']]
categorical  ::: SelectColumns = PortBankL[['age', 'duration', 'emp_var_rate']]
dummies = pd.get_dummies(categorical)
dummies.to_csv("dummies.csv", index=False) / dummies = pd.DataFrame(dummies) & numerical = pd.DataFrame(numerical)
dummies = pd.read_csv("dummies.csv")
PortBank1 = numerical.join(categorical)
PortBank1 = pd.DataFrame(PortBank1)
x = PortBank1.drop('y', axis=1)
y = PortBank1['y']
trainx, testx, trainy, testy = train_test_split(x, y, size=.3, random_state=10)
regr = linear_model.LinearRegression()
regr.fit(trainx, trainy)
predicty = regr.predict(testx)

# Metrics

# Linear Regression

print('Coefficients: \n', regr.coef_)
print('Mean square error: %.2f' % mean_squared_error(testy, predicty))
print('Variance: %.2f' % r2_score(testy, predicty))

# RSquared / Accuracy

# Linear Regression

regr.score(x, y)


# Visualization

# Histogram

iris.sepal_length.hist()
plt.show()

# Scatter Plot

fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
plt.show()
--------------------------------------------------------------------------------

