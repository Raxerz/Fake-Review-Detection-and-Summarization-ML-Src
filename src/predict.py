# Load libraries
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

newpath = ML_GENERATED_CSV_PATH + '/musical_label.csv'
cols = pandas.read_csv(newpath,header=None).columns
X_validation = pandas.read_csv(newpath,header=None,usecols=cols[1:])
scoring = 'accuracy'

print("1. K Nearest Neighbours")
print("2. SVM")
print("3. Decision Trees")
print("4. Naive Bayes")
print("5. Linear Discriminant Analysis")
print("Enter your choice")
ch=int(raw_input())
if ch==1:
	knn = pickle.load(open( ML_MODELS_PATH + "/knn.p", "rb" ))
	predictions = knn.predict(X_validation)
	with open(newpath) as fin, open('knn.csv', 'w') as fout:
		index = 0
		for line in iter(fin.readline, ''):
		    fout.write(line.replace('\n', ',' + str(predictions[index]) + '\n'))
		    index += 1

	outputpath = ML_GENERATED_CSV_PATH + '/musical_kmeans_label.csv'
	cols = pandas.read_csv(outputpath,header=None, nrows=1).columns
	Y_validation = np.array(pandas.read_csv(outputpath,header=None,usecols=cols[len(cols)-1:])).ravel()
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
elif ch==2:
	svc = pickle.load(open( ML_MODELS_PATH + "/svc.p", "rb" ))
	predictions = svc.predict(X_validation)
	with open(newpath) as fin, open('svc.csv', 'w') as fout:
		index = 0
		for line in iter(fin.readline, ''):
		    fout.write(line.replace('\n', ',' + str(predictions[index]) + '\n'))
		    index += 1

	outputpath = ML_GENERATED_CSV_PATH + '/musical_kmeans_label.csv'
	cols = pandas.read_csv(outputpath,header=None, nrows=1).columns
	Y_validation = np.array(pandas.read_csv(outputpath,header=None,usecols=cols[len(cols)-1:])).ravel()
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
elif ch==3:
	cart = pickle.load(open( ML_MODELS_PATH + "/cart.p", "rb" ))
	predictions = cart.predict(X_validation)
	with open(newpath) as fin, open('cart.csv', 'w') as fout:
		index = 0
		for line in iter(fin.readline, ''):
		    fout.write(line.replace('\n', ',' + str(predictions[index]) + '\n'))
		    index += 1

	outputpath = ML_GENERATED_CSV_PATH + '/musical_kmeans_label.csv'
	cols = pandas.read_csv(outputpath,header=None, nrows=1).columns
	Y_validation = np.array(pandas.read_csv(outputpath,header=None,usecols=cols[len(cols)-1:])).ravel()
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
elif ch==4:
	nb = pickle.load(open( ML_MODELS_PATH + "/nb.p", "rb" ))
	predictions = nb.predict(X_validation)
	with open(newpath) as fin, open('nb.csv', 'w') as fout:
		index = 0
		for line in iter(fin.readline, ''):
		    fout.write(line.replace('\n', ',' + str(predictions[index]) + '\n'))
		    index += 1

	outputpath = ML_GENERATED_CSV_PATH + '/musical_kmeans_label.csv'
	cols = pandas.read_csv(outputpath,header=None, nrows=1).columns
	Y_validation = np.array(pandas.read_csv(outputpath,header=None,usecols=cols[len(cols)-1:])).ravel()
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
elif ch==5:
	lda = pickle.load(open( ML_MODELS_PATH + "/lda.p", "rb" ))
	predictions = lda.predict(X_validation)
	with open(newpath) as fin, open('lda.csv', 'w') as fout:
		index = 0
		for line in iter(fin.readline, ''):
		    fout.write(line.replace('\n', ',' + str(predictions[index]) + '\n'))
		    index += 1

	outputpath = ML_GENERATED_CSV_PATH + '/musical_kmeans_label.csv'
	cols = pandas.read_csv(outputpath,header=None, nrows=1).columns
	Y_validation = np.array(pandas.read_csv(outputpath,header=None,usecols=cols[len(cols)-1:])).ravel()
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))
