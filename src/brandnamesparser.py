import re
import os
import json
import gzip
import pickle
import os.path
import numpy as np
from constants import *
from numpy import savez_compressed, load

DEBUG = False

def parse(path, domain):
	writepath = BRANDS_PARSED_PATH + '/' + domain.lower() + '.npz'
	if not os.path.exists(writepath):
		g = gzip.open(path, 'r')
		brands = {}
		brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
		brands_reviews = brands_reviews['arr_0'].tolist()
		for l in g:
			json_data = json.dumps(eval(l))
			data = re.match(r'(\{.*})',json_data)
			json_obj = json.loads(data.group())
			prod = {}
			if "brand" in json_obj and json_obj["brand"]!="" and "title" in json_obj and json_obj["asin"] in brands_reviews.keys():
				prod[json_obj["title"]] = json_obj["asin"]
				if json_obj["brand"] not in brands:
					brands[json_obj["brand"]] = []
				brands[json_obj["brand"]].append(prod)
		savez_compressed(writepath, brands)
		print("Brand names extracted successfully for " + domain.lower())
	else:
		if DEBUG:
			print("Brand names are parsed and available for " + domain.lower())

if __name__ == '__main__':
	extractBrandNames()

def extractBrandNames():
	for file in os.listdir(BRANDS_RAW_PATH):
		if file.endswith(".json.gz"):
			parse(os.path.join(BRANDS_RAW_PATH, file), file.split('.json.gz')[0])
