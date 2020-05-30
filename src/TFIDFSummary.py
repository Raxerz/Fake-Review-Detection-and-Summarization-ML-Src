import os
import re
import nltk
import math
import pickle
from time import sleep
from constants import *
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams, ngrams
from numpy import savez_compressed, load

def summaryGen(fileName,domain,gram=5,debug=False):
	if os.path.exists(REVIEWS_PARSED_PATH + '/' + domain.lower()+".npz"):
		stopwords = nltk.corpus.stopwords.words()
		tokenizer = RegexpTokenizer("[\w']+", flags=re.UNICODE)

		def freq(word, doc):
		    return doc.count(word)


		def word_count(doc):
		    return len(doc)


		def tf(word, doc):
		    return (freq(word, doc) / float(word_count(doc)))


		def num_docs_containing(word, list_of_docs):
			count = 0
			for document in list_of_docs:
				if freq(word, document) > 0:
					count += 1
			return 1 + count


		def idf(word, list_of_docs):
		    return math.log(len(list_of_docs) /
			    float(num_docs_containing(word, list_of_docs)))


		def tf_idf(word, doc, list_of_docs):
		    return (tf(word, doc) * idf(word, list_of_docs))

		#Compute the frequency for each term.
		vocabulary = []
		docs = {}
		all_tips = []
		text = ""
		brands_reviews = load(REVIEWS_PARSED_PATH + '/' + domain.lower() + '.npz', allow_pickle=True)
		brands_reviews = brands_reviews['arr_0'].tolist()
		review_data = brands_reviews[fileName]
		for i in review_data:
			text+=i["review"]

		tokens = tokenizer.tokenize(text)

		bi_tokens = bigrams(tokens)
		tri_tokens = trigrams(tokens)
		n_tokens = ngrams(tokens, gram)
		tokens = [token.lower() for token in tokens if len(token) > 2]
		tokens = [token for token in tokens if token not in stopwords]

		bi_tokens = [' '.join(token).lower() for token in bi_tokens]
		bi_tokens = [token for token in bi_tokens if token not in stopwords]

		tri_tokens = [' '.join(token).lower() for token in tri_tokens]
		tri_tokens = [token for token in tri_tokens if token not in stopwords]

		n_tokens = [' '.join(token).lower() for token in n_tokens]
		n_tokens = [token for token in n_tokens if token not in stopwords]

		final_tokens = []
		final_tokens.extend(tokens)
		final_tokens.extend(bi_tokens)
		final_tokens.extend(tri_tokens)
		final_tokens.extend(n_tokens)
		docs[0] = {'freq': {}, 'tf': {}, 'idf': {},
				'tf-idf': {}, 'tokens': []}

		for token in final_tokens:
			#The frequency computed for each tip
			docs[0]['freq'][token] = freq(token, final_tokens)
			#The term-frequency (Normalized Frequency)
			docs[0]['tf'][token] = tf(token, final_tokens)
			docs[0]['tokens'] = final_tokens

		vocabulary.append(final_tokens)

		for doc in docs:
			for token in docs[doc]['tf']:
				#The Inverse-Document-Frequency
				docs[doc]['idf'][token] = idf(token, vocabulary)
				#The tf-idf
				docs[doc]['tf-idf'][token] = tf_idf(token, docs[doc]['tokens'], vocabulary)

		#Now let's find out the most relevant words by tf-idf.
		words = {}
		for doc in docs:
			for token in docs[doc]['tf-idf']:
				if token not in words:
					words[token] = docs[doc]['tf-idf'][token]
				else:
					if docs[doc]['tf-idf'][token] > words[token]:
						words[token] = docs[doc]['tf-idf'][token]

		review_keywords = sorted(words.items(), key=lambda x: x[1], reverse=True)
		if debug:
			print("After tokenization....")
			sleep(1)
			print(final_tokens)
			sleep(1)
			print("After frequency computation....")
			sleep(1)
			print(docs[0]['freq'])
			sleep(1)
			print("After term frequency computation....")
			sleep(1)
			print(docs[0]['tf'])
			sleep(1)
			print("After Inverse-Document-Frequency computation....")
			sleep(1)
			print(docs[0]['tf-idf'])
			sleep(1)
			print("After term-frequency Inverse-Document-Frequency computation....")
			sleep(1)
			print(docs[0]['tf-idf'])
		sleep(1)
		print("Scores.....")
		for i in review_keywords:
			print(i[0],"-",i[1])
		return [i[0] for i in review_keywords]
	else:
		print("Domain not in dataset")
		return "Domain not in dataset"
