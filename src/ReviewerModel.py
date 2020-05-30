# Load libraries
import os
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np
import pickle

def evaluate(model,domain):
	path = ML_GENERATED_CSV_PATH + '/cellphones_label_noheader.csv'
	newpath = ML_GENERATED_CSV_PATH + '/'+domain.lower()+'.csv'
	outputpath = ML_GENERATED_CSV_PATH + '/'+domain.lower()+'_label.csv'
	if not os.path.exists(path) or not os.path.exists(newpath) or not os.path.exists(outputpath):
		return "One or more files do not exist"

	cols = pandas.read_csv(path, header=None).columns
	X = pandas.read_csv(path,header=None,usecols=cols[1:len(cols)-1])
	Y = np.array(pandas.read_csv(path,header=None,usecols=cols[len(cols)-1:])).ravel()
	cols = pandas.read_csv(newpath).columns
	X_test = pandas.read_csv(newpath,usecols=cols[1:])
	scoring = 'accuracy'
	X_train, X_validation, Y_train = X, X_test, Y
	seed = 7
	'''models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	# evaluate each model in turn
	results = []
	names = []
	for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)'''

	# Make predictions on validation dataset
	if model == "knn":
		knn = KNeighborsClassifier()
		knn.fit(X_train, Y_train)
		predictions = knn.predict(X_validation)

	elif model == "cart":
		cart = DecisionTreeClassifier()
		cart.fit(X_train, Y_train)
		predictions = cart.predict(X_validation)

	elif model == "svm":
		svc = SVC(kernel='linear')
		svc.fit(X_train, Y_train)
		predictions = svc.predict(X_validation)

	elif model == "nb":
		nb = GaussianNB()
		nb.fit(X_train, Y_train)
		predictions = nb.predict(X_validation)

	elif model == "lda":
		lda = LinearDiscriminantAnalysis(n_components=2)
		lda.fit(X_train, Y_train)
		predictions = lda.predict(X_validation)

	cols = pandas.read_csv(outputpath, nrows=1).columns
	Y_validation = np.array(pandas.read_csv(outputpath,usecols=cols[len(cols)-1:])).ravel()
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
	return accuracy_score
