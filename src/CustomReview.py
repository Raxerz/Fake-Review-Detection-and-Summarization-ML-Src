import re
import os
import time
import gzip
import nltk
import json
import pandas
import pickle
import string
import keywords
import collections
import numpy as np
from constants import *
from nltk import tokenize
from textblob import TextBlob
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
import warnings
from numpy import savez_compressed, load

warnings.filterwarnings("ignore", category=DeprecationWarning)
fakeness = 0.0
def parse_custom(domain="cellphones"):
	global fakeness
	brands = load(BRANDS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
	brands = brands['arr_0'].tolist()
	prodslist={}
	c=0;
	brandslist={}
	prodslist={}
	for brand in brands.keys():
		brandslist[c+1]=brand
		print(str(c+1)+". "+brand+"\n")
		c+=1

	print("Enter your choice")
	ch=int(input())
	selectedBrand = brandslist[ch]
	c=0
	for prods in range(len(brands[selectedBrand])):
		for prod in brands[selectedBrand][prods].keys():
			prodslist[c+1]=brands[selectedBrand][prods][prod]
			print(str(c+1)+". "+prod+"\n")
		c+=1
	print("Enter your choice")
	ch=int(input())
	review = ""
	print("Enter your review")
	while True:
		try:
			line = input("")
		except EOFError:
			break
		review+=line
	print("\nPlease wait a moment. Processing the result...\n")
	keywords_list = keywords.extract_keywords(domain,prodslist[ch])
	stopwords = nltk.corpus.stopwords.words()
	brandslist = []
	for i in brands.keys():
		brand_words = i.split()
		brandslist+=brand_words
		for j in brands[i]:
			title_words = j.items()[0][0].split()
			brandslist+=title_words
	brandslist = [token for token in brandslist if token not in stopwords]
	brandslist = set(brandslist)
	vocabulary = []
	reviewList = []
	stopwords = nltk.corpus.stopwords.words()
	tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)
	text = nltk.word_tokenize(review)
	cnt = 0
	keyword_cnt = 0
	for i in text:
		if i in brandslist:
			cnt+=1
		if i in keywords_list:
			keyword_cnt += 1
	pos_tagged = nltk.pos_tag(text)
	analyze_text = TextBlob(review)
	counts = Counter(tag for word,tag in pos_tagged)
	caps = len(filter(lambda x: x in string.uppercase, review))
	review_status = [0 for i in range(7)]
	review_data = [0 for i in range(7)]
	if len(review)!=0:
		c = Counter(c for c in review if c in ["?","!"])
		review_data[0] = float(counts['PRP$'])/float(len(analyze_text.words))
		review_data[1] = analyze_text.subjectivity
		review_data[2] = float(caps)/len(review)
		review_data[3] = float(c["?"]+c["!"])/len(review)
		review_data[4] = float(len(analyze_text.words))/1000
		review_data[5] = float(cnt)/float(len(analyze_text.words))
		review_data[6] = float(keyword_cnt)/float(len(analyze_text.words))
	if review_data[0]>0:
		review_status[0] = 1
	if review_data[1] < 0.5:
		review_status[1] = 1
	if review_data[2] >= 0.5:
		review_status[2] = 1
	if review_data[3] >=0.1:
		review_status[3] = 1
	if review_data[4] <= 0.135:
		review_status[4] = 1
	if review_data[5] >= 0.5 or review_data[5]<=0.1:
		review_status[5] = 1
	if review_data[6] < 0.5:
		review_status[6] = 1
	detection_counter=collections.Counter(review_status)
	deceptive_level = (float(detection_counter[1])/7) * 100
	fakeness = deceptive_level
	return review_data

def parse(readpath = REVIEWS_RAW_PATH + '/cellphones.json.gz',writepath = ML_GENERATED_CSV_PATH + '/custom_review.csv', domain="cellphones"):
	brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
	brands_reviews = brands_reviews['arr_0'].tolist()
	stopwords = nltk.corpus.stopwords.words()
	docs={}
	brands = load(BRANDS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
	brands = brands['arr_0'].tolist()
	brandslist = []
	for i in brands.keys():
		brand_words = i.split()
		brandslist+=brand_words
		for j in brands[i]:
			title_words = j.items()[0][0].split()
			brandslist+=title_words
	brandslist = [token for token in brandslist if token not in stopwords]
	brandslist = set(brandslist)
	vocabulary = []
	reviewList = []
	stopwords = nltk.corpus.stopwords.words()
	tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)
	g = gzip.open(readpath, 'r')
	fwrite = open(writepath,'w')
	reviewmap = {}
	reviewcount=0
	for l in g:
		if reviewcount <= 5000:
			cnt = 0
			keyword_cnt = 0
			json_data = json.dumps(eval(l))
			json_obj = re.match(r'(\{.*})',json_data)
			data = json.loads(json_obj.group())
			data = l
			review = data["reviewText"]
			asin = data["asin"]
			keywords_list = keywords.extract_keywords_custom("cellphones",asin,brands_reviews)
			review_data = brands_reviews[asin]
			text = nltk.word_tokenize(review)
			for i in text:
				if i in brandslist:
					cnt += 1
				if i in keywords_list:
					keyword_cnt += 1
			pos_tagged = nltk.pos_tag(text)
			analyze_text = TextBlob(review)
			counts = Counter(tag for word,tag in pos_tagged)
			caps = len(filter(lambda x: x in string.uppercase, review))
			review_status = [0 for i in range(7)]
			review_data = [0 for i in range(7)]
			if len(review)!=0:
				c = Counter(c for c in review if c in ["?","!"])
				review_data[0] = float(counts['PRP$'])/float(len(analyze_text.words))
				review_data[1] = analyze_text.subjectivity
				review_data[2] = float(caps)/len(review)
				review_data[3] = float(c["?"]+c["!"])/len(review)
				review_data[4] = float(len(analyze_text.words))/1000
				review_data[5] = float(cnt)/float(len(analyze_text.words))
				review_data[6] = float(keyword_cnt)/float(len(analyze_text.words))
				if review_data[0]>0:
					review_status[0] = 1
				if review_data[1] < 0.5:
					review_status[1] = 1
				if review_data[2] >= 0.5:
					review_status[2] = 1
				if review_data[3] >=0.1:
					review_status[3] = 1
				if review_data[4] <= 0.135:
					review_status[4] = 1
				if review_data[5] >= 0.5 or review_data[5]<=0.1:
					review_status[5] = 1
				if review_data[6] < 0.2:
					review_status[6] = 1
				detection_counter=collections.Counter(review_status)
				if(review_data[6] < 0.2 or detection_counter[1]>=4):
					label = "FAKE"
				else:
					label = "TRUTHFUL"
				fwrite.write(str(review_status[0])+","+str(review_status[1])+","+str(review_status[2])+","+str(review_status[3])+","+str(review_status[4])+","+str(review_status[5])+","+str(review_status[6])+","+label+"\n")
			reviewcount+=1
		else:
			break

def predict(domain="cellphones"):
	brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
	brands_reviews = brands_reviews['arr_0'].tolist()
	newpath = ML_GENERATED_CSV_PATH + '/custom_review.csv'
	cols = pandas.read_csv(newpath, header = None).columns
	X = pandas.read_csv(newpath,header = None,usecols=cols[0:7])
	Y = np.array(pandas.read_csv(newpath, header = None,usecols=cols[7:8])).ravel()
	validation_size = 0.20
	seed = 7
	scoring = 'accuracy'
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	# Make predictions on validation dataset
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	pickle.dump( knn, open( ML_MODELS_PATH + "/save.p", "wb" ) )
	predictions = knn.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))

def pred():
	global fakeness
	print("1.CellPhones and Accessories")
	print("2.Movies and TV shows")
	print("3.Electronics")
	print("4.Clothes-and-Accessories")
	print("5.Video Games")
	print("6.Beauty Products")
	print("7.Automotive")
	print("8.Musical-Instruments")
	print("Enter your choice")
	dom_choice=int(input())

	domain_list=["CellPhones","Movies&TV","Electronics","Clothes&Acc","VideoGames","Beauty","Automotive","Musical-Instruments"]
	knn = pickle.load(open( ML_MODELS_PATH + "/save.p", "rb" ))
	X_validation = parse_custom(domain_list[dom_choice-1])
	predictions = knn.predict(X_validation)
	print("This review is " + str(fakeness) + "% fake")
	'''if predictions[0]=="TRUTHFUL" and fakeness>=40:
		print "Overall : This is a FAKE review"
	elif predictions[0]=="FAKE" and fakeness<40:
		print "Overall : This is a TRUTHFUL review"
	else:
		print "Overall : This is a " + predictions[0] + " review"'''
if __name__=='__main__':
	if not os.path.exists(ML_GENERATED_CSV_PATH + '/custom_review.csv'):
		parse()
	if not os.path.exists(ML_MODELS_PATH + '/save.p'):
		predict()
	else:
		pred()
