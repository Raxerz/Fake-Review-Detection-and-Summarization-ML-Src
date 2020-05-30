import re
import csv
import json
import gzip
import pickle
import os.path
import numpy as n
import ReviewBased
import pandas as pd
import ReviewerBased
import CosineSimilarity
from constants import *
import ReviewerBasedPreprocess
from numpy import savez_compressed, load

print("1.CellPhones and Accessories")
print("2.Movies and TV shows")
print("3.Android Apps")
print("4.Clothes-and-Accessories")
print("5.Video Games")
print("6.Beauty Products")
print("7.Automotive")
print("8.Musical-Instruments")
print("9.Electronics")

print("Enter your choice")
dom_choice=int(input())
domain_list=["CellPhones","Movies&TV","AndroidApps","Clothes&Acc","VideoGames","Beauty","Automotive","Musical-Instruments","Electronics"]

print("1.Reviewer Based")
print("2.Review Based")
print("3.Cosine similarity")
print("Enter your choice")
choice=int(input())

def reviewerBasedPrint(domain):
	fwo = open(GOLD_REVIEWERS,"r")
	goldstdreviewers = fwo.read().strip().split("\n")
	if domain.lower()!="androidapps":
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
		brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
		brands_reviews = brands_reviews['arr_0'].tolist()
		f = open(ML_GENERATED_CSV_PATH + '/' + domain.lower()+'_label.csv', 'rt')
		try:
			reader = csv.reader(f)
			for row in reader:
				if row[7]=="FAKE":
					for i in brands_reviews[prodslist[ch]]:
						if i["reviewerID"] == row[0] and row[0] not in goldstdreviewers:
							print("ReviewerID : ",row[0])
							print("Rating for this product : ",i["score"])
							print("Helpfulness of this review : ",i["helpful"])
							print("Negative rating by reviewer : ",row[1])
							print("Neutral rating by reviewer : ",row[2])
							print("Positive rating by reviewer : ",row[3])
							print("Helpfulness ratio of reviewer : ",row[4], " people found the reviewer useful")
							print("Burst intensity of reviewer : ",row[5]," reviews every 2 days")
							print("Nuber of reviews posted : ",row[6])
							print("\nReview Text\n")
							print(i["review"])
							print("\n")
		finally:
		    f.close()
	else:
		g = load(REVIEWS_RAW_PATH + '/' + domain.lower() + '.json.gz', allow_pickle=True)
		df = pd.read_csv(ML_GENERATED_CSV_PATH + '/' + domain.lower()+"_label.csv")

		for j in df[df['Label'] == 'FAKE']["ReviewerID"].tolist():
			if j not in goldstdreviewers:
				print(j,"FAKE")
				for l in g:
					json_data = json.dumps(eval(l))
					json_obj = re.match(r'(\{.*})',json_data)
					data = json.loads(json_obj.group())
					if data["reviewerID"]==j:
						print(data["reviewText"])
						print("")
				print("")
		for j in df[df['Label'] == 'TRUTHFUL']["ReviewerID"].tolist():
			if j in goldstdreviewers:
				print(j,"TRUTHFUL")
		for j in df[df['Label'] == 'VERY TRUTHFUL']["ReviewerID"].tolist():
			if j in goldstdreviewers:
				print(j,"VERY TRUTHFUL")

def reviewerBased(domain):
	if os.path.exists(ML_GENERATED_CSV_PATH + '/' + domain.lower()+"_label.csv"):
		reviewerBasedPrint(domain)
	else:
		if not os.path.exists(ML_GENERATED_CSV_PATH + '/' + domain.lower()+'.csv'):
			print("Preprocessing...")
			ReviewerBasedPreprocess.parse(REVIEWS_RAW_PATH +'/' + domain.lower()+'.json.gz',ML_GENERATED_CSV_PATH + '/' + domain.lower()+'.csv')
		print("1. Use K-Means algorithm")
		print("2. Use Models")
		print("Enter your choice")
		algo = int(input())
		if algo==1:
			print("Training and evaluating...")
			ReviewerBased.evaluate(domain.lower(),ML_GENERATED_CSV_PATH + '/' + domain.lower()+'.csv')
		elif algo==2:
			print("1. K Nearest Neighbours")
			print("2. SVM")
			print("3. Decision Trees")
			print("4. Naive Bayes")
			print("5. Linear Discriminant Analysis")
			print("Choose classifier")
			cls = int(input())
			print("Predicting using trained model...")
			ReviewerBased.predict(cls,domain.lower())
			if os.path.exists(ML_GENERATED_CSV_PATH + '/' + domain.lower()+"_label.csv"):
				reviewerBasedPrint(domain)

def reviewBased(domain):
	domainPreprocessed = domain.replace('&', '_')
	if not os.path.exists(ML_GENERATED_CSV_PATH + '/' +domainPreprocessed.lower()+'_review.csv'):
		print("Preprocessing...")
		ReviewBased.parse(REVIEWS_RAW_PATH + '/' + domain.lower()+'.json.gz',ML_GENERATED_CSV_PATH + '/' +domain.lower()+'_review.csv',domain.lower())
	if not os.path.exists(ML_GENERATED_CSV_PATH + '/' + domainPreprocessed.lower()+'_review_label.csv'):
		print("Predicting using trained model...")
		print("1. K Nearest Neighbours")
		print("2. SVM")
		print("3. Decision Trees")
		print("4. Naive Bayes")
		print("5. Linear Discriminant Analysis")
		print("Choose classifier")
		cls = int(input())
		ReviewBased.predict(cls,domain.lower())
	else:
		f = open(ML_GENERATED_CSV_PATH + '/' +domainPreprocessed.lower()+'_review_label.csv', 'rt')
		try:
			reader = csv.reader(f)
			for row in reader:
				if row[8]=="FAKE":
					print("\n")
					print("ReviewerID : ",row[0])
					print("BrandID : ",row[1])
					print("Review")
					print("".join(row[9:]))
					print("\n")
		finally:
		    f.close()

if(dom_choice==1):
	domain=domain_list[0]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==2):
	domain=domain_list[1]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==3):
	domain=domain_list[2]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==4):
	domain=domain_list[3]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==5):
	domain=domain_list[4]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==6):
	domain=domain_list[5]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==7):
	domain=domain_list[6]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)

elif(dom_choice==8):
	domain=domain_list[7]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)
elif(dom_choice==9):
	domain=domain_list[8]
	if choice==1:
		reviewerBased(domain)
	elif choice==2:
		reviewBased(domain)
	elif(choice==3):
		CosineSimilarity.parse(domain)
else:
	print("Invalid choice")
