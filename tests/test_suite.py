import sys
sys.path.append("../src")
import unittest
import SummaryMain
import TextRank
import TFIDFSummary
import ReviewModel
import ReviewerModel

class TestSummaryMain(unittest.TestCase):
	
	def runTest(self):
		print "\nRunning SummaryMain test cases...\n"
		self.assertEqual( SummaryMain.main(5,[]), "Wrong choice")

class TextRankSummary(unittest.TestCase):

	def runTest(self):
		print "\nRunning TextRank Summary test cases...\n"
		self.assertEqual( TextRank.summaryGen("abc.txt","abc"), "Domain not in dataset")

class TestTFIDSummary(unittest.TestCase):
 
	def runTest(self):
		print "\nRunning TFIDF Summary test cases...\n"
		self.assertEqual( TFIDFSummary.summaryGen("abc.txt","abc"), "Domain not in dataset")

class TestReviewModel(unittest.TestCase):

	def runTest(self):
		print "\nRunning Review Based Fake Review Detection test cases...\n"
		self.assertEqual( ReviewModel.evaluate("abc","def"), "One or more files do not exist")
		print "KNN - REVIEW BASED"
		self.assertGreater(ReviewModel.evaluate("knn","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")
		print "CART - REVIEW BASED"
		self.assertGreater(ReviewModel.evaluate("cart","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")
		print "LDA - REVIEW BASED"
		self.assertGreater(ReviewModel.evaluate("lda","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")
		print "SVM - REVIEW BASED"
		self.assertGreater(ReviewModel.evaluate("svm","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")	
		print "NB - REVIEW BASED"
		self.assertGreater(ReviewModel.evaluate("nb","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")

class TestReviewerModel(unittest.TestCase):
 
	def runTest(self):
		print "\nRunning Reviewer Based Fake Review Detection test cases...\n"
		self.assertEqual( ReviewerModel.evaluate("abc","def"), "One or more files do not exist")
		print "KNN - REVIEWER BASED"
		self.assertGreater(ReviewerModel.evaluate("knn","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")
		print "CART - REVIEWER BASED"
		self.assertGreater(ReviewerModel.evaluate("cart","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")
		print "LDA - REVIEWER BASED"
		self.assertGreater(ReviewerModel.evaluate("lda","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")
		print "SVM - REVIEWER BASED"
		self.assertGreater(ReviewerModel.evaluate("svm","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")	
		print "NB - REVIEWER BASED"
		self.assertGreater(ReviewerModel.evaluate("nb","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")

def suite():

	suite = unittest.TestSuite()
	suite.addTest(TestSummaryMain())
	suite.addTest(TestTFIDSummary())
	suite.addTest(TextRankSummary())
	suite.addTest(TestReviewModel())
	suite.addTest(TestReviewerModel())

	return suite

if __name__ == '__main__':

	print "Running test suite..."
	log_file = 'logs/tests.log'

	f = open(log_file, "w")

	runner = unittest.TextTestRunner(f)

	test_suite = suite()

	runner.run (test_suite)

