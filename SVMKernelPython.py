

linear_svc = svm.SVC(kernel='linear')
linear_svc.kernel
'linear'

rbf_svc = svm.SVC(kernel='rbf')
rbf_svc.kernel
'rbf'

poly_svc = svm.SVC(kernel='polynomial')
poly_svc.kernel
'polynomial'

sig_svc = svm.SVC(kernel='sigmoid')
sig_svc.kernel
'sigmoid'