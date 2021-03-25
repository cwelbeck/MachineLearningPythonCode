

# Linear Regression  https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
#Linear Regression  http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# Linear Regression https://pythonspot.com/linear-regression/  pd.read_csv()
# Linear Regression https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb  pd.read_csv()
# Linear Regression * http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
# Multiple Linear Regression * https://www.datarobot.com/blog/multiple-regression-using-statsmodels/  **** StatsModel
# Logistic Regression https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
# Logistic Regression https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/
# Decision Tree https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
# Decision Tree http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
# Decision Tree https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# Decision Tree http://scikit-learn.org/stable/modules/tree.html
# Decision Tree * https://www.geeksforgeeks.org/decision-tree-implementation-python/
# Decision Tree ** http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
# Multiple Regression  https://www.datarobot.com/blog/multiple-regression-using-statsmodels/

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing, datasets, linear_model, metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

import os
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
# now, to clear the screen
cls()

# Files Directory
for x in os.listdir('C:/Users/ch373676/Downloads/Data Science Datasets'):
 print(x)

 # Set Working Directory
 import os
 os.chdir('C:/Users/ch373676/Downloads/Data Science Datasets')
 
 # Print Working Directory
 import os;
print os.getcwd();

# Files in Current Directory
os.listdir()
 
# Load CVS

data = pd.read_csv('bank.csv', header=0)
data = data.dropna()
data = data[data.latitude > 0]
data.drop_duplicates(inplace = True)
data = data[data.sq__ft> 0]
print(data.shape)
print(list(data.columns))

df = pd.read_csv('pandas_dataframe_importing_csv/example.csv')
df

# Load Datasets from sklearn
from sklearn import datasets
diabetes = datasets.load_diabetes()
diabetes.keys()
dict_keys(['data', 'target', 'DESCR', 'feature_names'])
df_x = pd.DataFrame(diabetes.feature_names)
df_x
df_y = pd.DataFrame(diabetes.target)
df_y

# define the data/predictors as the pre-set feature names    **********
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame  *********
target = pd.DataFrame(data.target, columns=["MEDV"])

## Preprocessing

# Selection Choose Rearrange Columns
SelectColumns = PortBankL[['age', 'duration', 'emp_var_rate']]
x = PortBank.drop('y', axis=1)
y = PortBank['y']
x = PortBank.iloc[:, 1:15]
y = PortBank.iloc[:, 0]
x = PortBank.values[:, 1:15]
y = PortBank.values[:, 0]
x = df.ix[0, 'Col1':'Col5']
x = data.iloc[:, :-1].values
y = data.iloc[:, 3].values
columns = iris[['sepal_length', 'sepal_width', 'petal_length']]
lm = smf.ols(formula='petal_width ~ columns', data=iris).fit()
lm.rsquared

# Drop Columns
from sklearn import preprocessing
x = PortBank.drop('y', axis=1)
PortBank.drop(PortBank.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)
x = PortBank.drop('y', axis=1)

# Create Dummy Variables for Linear/Logistic Regression
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
dummy_columns = iris[['sepal_length', 'sepal_width', 'petal_length']]
dummies = pd.get_dummies(dummy_columns)
iris2.join(dummies)

# pd.read_csv for Linear Regression
PortBank = pd.read_csv('PortBank.csv')
PortBank.to_csv('PortBank.csv', index=False)
from sklearn.feature_extraction import DictVectorizer
vectorizer = DictVectorizer()
vector_data = vectorizer.fit_transform(PortBank)
PortBank1 = pd.DataFrame(PortBank1) ***
x = PortBank.drop('y', axis=1)
y = PortBank['y']
x = x.apply(pd.to_numeric, errors='coerce')
y = y.apply(pd.to_numeric, errors='coerce')
x = PortBank.iloc[:, :-1].values
y = PortBank.iloc[:, 3].values
trainx = x[:-20]
testx = x[-20:]
trainy = y[:-20]
testy = y[-20:]

# Train & Split Dataset
trainx , testx, trainy, testy = train_test_split(x, y, test_size=.3, random_state=10)
regr.fit(trainx, trainy)


# Data Details
data.head(5)
data.tail()

# Check for Missing Data
pd.isnull(data)
pd.isnull(data).sum()
pd.isnull(PortBank).sum()
PortBank.isnull().sum()

# count the number of NaN values in each column
print(dataset.isnull().sum())

# Drop Missing Data
dataset.dropna()

#  # Impute Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3]) # only fit columns 1 and 2
x[:, 1:3] = imputer.transform(x[:, 1:3])  # imputes the mean for the missing data

#  # Converting Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Replace Missing Data with numpy.NaN)
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
dataset.replace(0, numpy.NaN)



##  Convert Categorical to Numerical as Factor

# encode df.famhist as a numeric via pd.Factor
df['famhist_ord'] = pd.Categorical(df.famhist).labels

est = smf.ols(formula="chd ~ famhist_ord", data=df).fit()

data['Salary'] = data['Salary'].apply(pd.to_numeric, errors='coerce')

## Convert DataFrame to Numpy Array
DataFrame.as_matrix(columns=None)[source]
x = pd.DataFrame(data=x)
y = pd.DataFrame(data=y)
x = x.as_matrix(columns=None)
y = y.as_matrix(columns=None)

# Rearrange Columns

# Data Exploration Count Value in column / Count Values in each Column
PortBank['y'].value_counts()

# Group by (Y) (0/1)
PortBank.groupby('y').mean()

# Create Dummy Variables
PortBank = pd.get_dummies(PortBank, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])

# Feature Scaling



#Visualization Plot Barplot
import seaborn as sns
PortBank['y'].value_counts()
sns.countplot(y='job', data=PortBank)
plt.show()
# Bar plot
sns.countplot(x='y', data=PortBank, palette='hls')
plt.show()
sns.countplot(x="marital", data=data)
plt.show()
plt.savefig('count_plot')

#Visualization Plot Barplot 2
pd.crosstab(PortBank.job, PortBank.y).plot(kind='Bar')
plt.title('Frequency of Purchase Per Job')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.show()

#Visualization Plot Stacked Barplot

table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')


# Barplot for the dependent variable (y)
sns.countplot(x='job', data=PortBank, palette='hls')


# Histogram
PortBank.age.hist()
plt.title("Histogram of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Visualize the relationship between the features and the response using scatterplots (Linear Regression)

fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])
plt.show()

------------ Linear Regression Plot
fig, ax = plt.subplots()
fit = np.polyfit(x, y, deg=1)
ax.plot(x, fit[0] * x + fit[1], color='red')
ax.scatter(x, y)
fig.show()


# Data Preprocessing

# Feature Selection  https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )
print(rfe.support_)
print(rfe.ranking_)

# Feature Importance  http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html


# # Metrics

# Linear Regression Metrics

print('Coefficients: \n', regr.coef_)
print('Mean square error: %.2f' % mean_squared_error(testy, predicty))
print('Variance: %.2f' % r2_score(testy, predicty))

# Logistic Regression Metrics

print(metrics.classification_report(testy, predicty))
print(metrics.confusion_matrix(testy, predicty))

# RSquared / Accuracy

# Linear Regression RSquared / Accuracy

regr.score(x, y)


# Linear Regression Accuracy Score
print('Accuracy: {:.2f}'.format(regr.score(testx, testy)))
print('Accuracy: {:.2f}'.format(regr.score(testx, testy) * 100))
regr.score(x,y)
# Linear Regression Coefficients
regr.coef_

# Linear Regression InterceptLog
regr.intercept_

# Logistic Regression Accuracy Score
logreg.score(testx, testy)
accuracy_score(testy, predicty)
# Logistic Regression Accuracy Score


# Decision Tree Accuracy Score
print "Accuracy is ", accuracy_score(y_test,y_pred)*100
cal_accuracy(y_test, y_pred)
# Decision Tree Accuracy Score


# Save dataframe as csv in the working director
PortBank.to_csv('PortBank.csv', index=False)
