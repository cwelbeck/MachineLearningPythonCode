
# http://scikit-learn.org/stable/modules/svm.html

from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]

svm = svm.SVC()
svm.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
	
svm.predict([[2., 2.]])

# get support vectors
svm.support_vectors_

# get indices of support vectors
svm.support_

# get number of support vectors for each class
svm.n_support_


----------------------------------------------------------------------------------------
MULTI-CLASS
----------------------------------------------------------------------------------------

X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes: 4*3/2 = 6

clf.decision_function_shape = "ovr"
dec = clf.decision_function([[1]])
dec.shape[1] # 4 classes

lin_clf = svm.LinearSVC()
lin_clf.fit(X, Y) 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
dec = lin_clf.decision_function([[1]])
dec.shape[1]


