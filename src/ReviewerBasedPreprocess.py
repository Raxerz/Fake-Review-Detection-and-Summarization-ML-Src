import re
import os
import json
import gzip

def parse(readpath,writepath):
	g = gzip.open(readpath, 'r')
	brand_reviews = {}
	ff = open(writepath,'w+')
	reviewmap = {}
	for l in g:
		json_data = json.dumps(eval(l))
		json_obj = re.match(r'(\{.*})',json_data)
		data = json.loads(json_obj.group())
		if data['reviewerID'] not in reviewmap:
			reviewmap[data['reviewerID']] = {'helpful':0,'negative':0.0,'count':0,'time':[],'burst':0.0,'positive':0.0,'neutral':0.0}
		helpfulness = float(data['helpful'][0])/float(data['helpful'][1]) if data['helpful'][1] else 0
		if reviewmap[data['reviewerID']]['helpful']:
			reviewmap[data['reviewerID']]['helpful'] += helpfulness
		else:
			reviewmap[data['reviewerID']]['helpful'] = helpfulness
		if data['overall']==1.0 or data['overall']==2.0:
			reviewmap[data['reviewerID']]['negative'] += 1
		if data['overall']==3.0:
			reviewmap[data['reviewerID']]['neutral'] += 1
		if data['overall']==4.0 or data['overall']==5.0:
			reviewmap[data['reviewerID']]['positive'] += 1
		if reviewmap[data['reviewerID']]['time']:
			reviewmap[data['reviewerID']]['time'].append(data['unixReviewTime'])
		else:
			reviewmap[data['reviewerID']]['time'].append(data['unixReviewTime'])
		reviewmap[data['reviewerID']]['count']+=1

	ff.write("reviewerID,negative,neutral,positive,helpfulness,burst,count\n")
	for i in reviewmap:
		reviewmap[i]['negative']/=reviewmap[i]['count']
		reviewmap[i]['neutral']/=reviewmap[i]['count']
		reviewmap[i]['positive']/=reviewmap[i]['count']
		reviewmap[i]['time']=sorted(reviewmap[i]['time'])
		days = 1
		reviews_per_day = 0
		for j in range(len(reviewmap[i]['time'])-1):
			if abs(reviewmap[i]['time'][j]-reviewmap[i]['time'][j+1])<172800:
				reviews_per_day += 1
			else:
				days+=1

		reviewmap[i]['burst']=float(reviews_per_day)/float(days)
		ff.write(i+","+str(reviewmap[i]['negative'])+","+str(reviewmap[i]['neutral'])+","+str(reviewmap[i]['positive'])+","+str(reviewmap[i]['helpful'])+","+str(reviewmap[i]['burst'])+","+str(reviewmap[i]['count']))
		ff.write("\n")
