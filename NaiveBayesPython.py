

# https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/

#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

#Create a Gaussian Classifier
nb = GaussianNB()

# Train the model using the training sets 
nb.fit(x, y)

#Predict Output 
predicted= nb.predict([[1,2],[3,4]])
print predicted