
from sklearn.svm import SVR
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X, y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
	

# #############################################################################
# #############################################################################

print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# #############################################################################
# #############################################################################


import csv 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

with open('test.csv', 'r') as f1:
  train_dataframe = pd.read_csv(f1)

X_train = train_dataframe.iloc[:30,(0)]
y_train = train_dataframe.iloc[:30,(1)]

with open('test.csv','r') as f2:
     test_dataframe = pd.read_csv(f2)

X_test = test_dataframe.iloc[30:,(0)]
y_test = test_dataframe.iloc[30:,(1)]

svr = svm.SVR(kernel="rbf", gamma=0.1)
log = LinearRegression()
svr.fit(X_train.reshape(-1,1),y_train)
log.fit(X_train.reshape(-1,1), y_train)

predSVR = svr.predict(X_test.reshape(-1,1))
predLog = log.predict(X_test.reshape(-1,1))

plt.plot(X_test, y_test, label='true data')
plt.plot(X_test, predSVR, 'co', label='SVR')
plt.plot(X_test, predLog, 'mo', label='LogReg')
plt.legend()
plt.show()


# #############################################################################
# #############################################################################


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split

X = np.linspace(0,100,101)
y = np.array([(100*np.random.rand(1)+num) for num in (5*x+10)])

X_train, X_test, y_train, y_test = train_test_split(X, y)

svr = SVR(kernel='linear')
lm = LinearRegression()
svr.fit(X_train.reshape(-1,1),y_train.flatten())
lm.fit(X_train.reshape(-1,1), y_train.flatten())

pred_SVR = svr.predict(X_test.reshape(-1,1))
pred_lm = lm.predict(X_test.reshape(-1,1))

plt.plot(X,y, label='True data')
plt.plot(X_test[::2], pred_SVR[::2], 'co', label='SVR')
plt.plot(X_test[1::2], pred_lm[1::2], 'mo', label='Linear Reg')
plt.legend(loc='upper left');


# #############################################################################
# #############################################################################

Sample Code For Feature Scaling


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)


------------------------------------------------------------
# SVR EXAMPLE #
------------------------------------------------------------

from sklearn import svm
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = svm.SVR()
clf.fit(X, y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
clf.predict([[1, 1]])

