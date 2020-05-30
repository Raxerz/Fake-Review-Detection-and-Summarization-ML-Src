import re
import os
import time
import gzip
import nltk
import json
import pandas
import pickle
import string
import collections
from constants import *
from nltk import tokenize
from textblob import TextBlob
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import savez_compressed, load

def parse(readpath,writepath,domain):
	brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
	brands_reviews = brands_reviews['arr_0'].tolist()
	docs={}
	vocabulary = []
	reviewList = []
	stopwords = nltk.corpus.stopwords.words()
	tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)
	g = gzip.open(readpath, 'r')

	fwrite = open(writepath,'w')
	reviewmap = {}
	reviewcount=0
	for l in g:
		if reviewcount <= 10000:
			json_data = json.dumps(eval(l))
			json_obj = re.match(r'(\{.*})',json_data)
			data = json.loads(json_obj.group())
			rating = 0
			review = data["reviewText"]
			asin = data["asin"]
			review_data = brands_reviews[asin]
			for i in review_data:
				rating+=float(i["score"])
			average_rating = rating/float(len(review_data))
			caps = len(list(filter(lambda x: x in string.uppercase, review)))
			analyze_text = TextBlob(review)
			review_status = [0 for i in range(6)]
			review_data = [0 for i in range(6)]
			if len(review)!=0:# and abs(average_rating-float(data["overall"]))>2:
				c = Counter(c for c in review if c in ["?","!"])
				review_data[0] = abs(float(data["overall"])-average_rating)
				review_data[1] = analyze_text.subjectivity
				review_data[2] = float(caps)/len(review)
				review_data[3] = float(c["?"]+c["!"])/len(review)
				review_data[4] = float(len(analyze_text.words))/1000
				helpfulness = float(data['helpful'][0])/float(data['helpful'][1]) if data['helpful'][1] else 0
				review_data[5] = helpfulness
				fwrite.write(data["reviewerID"]+","+data["asin"]+","+str(review_data[0])+","+str(review_data[1])+","+str(review_data[2])+","+str(review_data[3])+","+str(review_data[4])+","+str(review_data[5])+"\n")
			reviewcount+=1
		else:
			break

def predict(ch,domain):
	brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
	brands_reviews = brands_reviews['arr_0'].tolist()
	domain = domain.replace('&', '_')
	newpath = ML_GENERATED_CSV_PATH + '/'+domain+'_review.csv'
	cols = pandas.read_csv(newpath).columns
	X_validation = pandas.read_csv(newpath,usecols=cols[2:8])
	scoring = 'accuracy'

	if ch==1:
		knn = pickle.load(open( ML_MODELS_PATH + "/knn_review.p", "rb" ))
		predictions = knn.predict(X_validation)
		with open(newpath) as fin, open(ML_GENERATED_CSV_PATH + '/'+domain+'_review_label.csv', 'w') as fout:
			index = 0
			for line in iter(fin.readline, ''):
				arr=line.split(",")
				reviewerID=arr[0]
				brandID=arr[1]
				for i in brands_reviews[brandID]:
					if i["reviewerID"]==reviewerID:
						review = i["review"]
				fout.write(line.replace('\n', ',' + str(predictions[index]) + ',' + review+'\n'))
				index += 1

	elif ch==2:
		svc = pickle.load(open( ML_MODELS_PATH + "/svc_review.p", "rb" ))
		predictions = svc.predict(X_validation)
		with open(newpath) as fin, open(ML_GENERATED_CSV_PATH + '/'+domain+'_review_label.csv', 'w') as fout:
			index = 0
			for line in iter(fin.readline, ''):
				arr=line.split(",")
				reviewerID=arr[0]
				brandID=arr[1]
				for i in brands_reviews[brandID]:
					if i["reviewerID"]==reviewerID:
						review = i["review"]
				fout.write(line.replace('\n', ',' + str(predictions[index]) + ',' + review+'\n'))
				index += 1

	elif ch==3:
		cart = pickle.load(open( ML_MODELS_PATH + "/cart_review.p", "rb" ))
		predictions = cart.predict(X_validation)
		with open(newpath) as fin, open(ML_GENERATED_CSV_PATH + '/'+domain+'_review_label.csv', 'w') as fout:
			index = 0
			header = fin.readline()
			for line in iter(fin.readline, ''):
				arr=line.split(",")
				reviewerID=arr[0]
				brandID=arr[1]
				for i in brands_reviews[brandID]:
					if i["reviewerID"]==reviewerID:
						review = i["review"]
				fout.write(line.replace('\n', ',' + str(predictions[index]) + ',' + review+'\n'))
				index += 1

	elif ch==4:
		nb = pickle.load(open( ML_MODELS_PATH + "/nb_review.p", "rb" ))
		predictions = nb.predict(X_validation)
		with open(newpath) as fin, open(ML_GENERATED_CSV_PATH + '/'+domain+'_review_label.csv', 'w') as fout:
			index = 0
			for line in iter(fin.readline, ''):
				arr=line.split(",")
				reviewerID=arr[0]
				brandID=arr[1]
				for i in brands_reviews[brandID]:
					if i["reviewerID"]==reviewerID:
						review = i["review"]
				fout.write(line.replace('\n', ',' + str(predictions[index]) + ',' + review+'\n'))
				index += 1

	elif ch==5:
		lda = pickle.load(open( ML_MODELS_PATH + "/lda_review.p", "rb" ))
		predictions = lda.predict(X_validation)
		with open(newpath) as fin, open(ML_GENERATED_CSV_PATH + '/'+domain+'_review_label.csv', 'w') as fout:
			index = 0
			for line in iter(fin.readline, ''):
				arr=line.split(",")
				reviewerID=arr[0]
				brandID=arr[1]
				for i in brands_reviews[brandID]:
					if i["reviewerID"]==reviewerID:
						review = i["review"]
				fout.write(line.replace('\n', ',' + str(predictions[index]) + ',' + review+'\n'))
				index += 1
