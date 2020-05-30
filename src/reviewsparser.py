import re
import os
import json
import gzip
import pickle
from constants import *
from numpy import savez_compressed, load

DEBUG = False

def parse(path, domain):
	writepath = REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz'
	if not os.path.exists(writepath):
		g = gzip.open(path, 'r')
		brand_reviews = {}
		reviewcount = 0
		for l in g:
			if(reviewcount <= 40000):
				json_data = json.dumps(eval(l))
				data = re.match(r'(\{.*})',json_data)
				json_obj = json.loads(data.group())
				obj = {"helpful":0.0,"score":0.0,"reviewerID":"","review":""}
				helpfulness = float(json_obj['helpful'][0]/json_obj['helpful'][1]) if json_obj['helpful'][1] else 0
				if "overall" in json_obj and "reviewerID" in json_obj and "reviewText" in json_obj:
					obj["helpful"] = helpfulness
					obj["score"] = json_obj["overall"]
					obj["reviewerID"] = json_obj["reviewerID"]
					obj["review"] = json_obj["reviewText"]
					if json_obj["asin"] not in brand_reviews:
						brand_reviews[json_obj["asin"]] = []
					brand_reviews[json_obj["asin"]].append(obj)
				reviewcount+=1
			else:
				break
		savez_compressed(writepath, brand_reviews)
		print("Reviews parsed successfully for " + domain.lower())
	else:
		if DEBUG:
			print("Reviews are parsed and available for " + domain.lower())

def parseReviews():
	for file in os.listdir(REVIEWS_RAW_PATH):
		if file.endswith(".json.gz"):
			parse(os.path.join(REVIEWS_RAW_PATH, file), file.split('.json.gz')[0])


if __name__ == '__main__':
	parseReviews()
