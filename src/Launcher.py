import SummaryLauncher

import reviewsparser
import brandnamesparser

reviewsparser.parseReviews()
brandnamesparser.extractBrandNames()

print("1. Summarization")
print("2. Fake review detection")
ch = int(input("Enter your choice\n"))
if ch==1:
	SummaryLauncher.summary()
elif ch==2:
	import FakeReviewDetection
elif ch==3:
	print("Invalid choice")
