import re
from constants import *
f = open(AMAZON_REVIEWERS_PATH,"r")
fw = open(GOLD_REVIEWERS,"w")
headers = f.readline()
for line in f:
	m = re.match(r'.*https://www.amazon.com/gp/pdp/profile/(?P<reviewerID>\w+)/.*', line)
	fw.write(m.group("reviewerID")+"\n")

fw.close()
