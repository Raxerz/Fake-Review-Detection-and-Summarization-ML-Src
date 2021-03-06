import sys
sys.path.append("../src")
import unittest
import ReviewerModel
 
class TestReviewerModel(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_reviewer_model_path(self):
        self.assertEqual( ReviewerModel.evaluate("abc","def"), "One or more files do not exist")

    def test_reviewer_knn_model(self):
        self.assertGreater(ReviewerModel.evaluate("knn","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")	

    def test_reviewer_cart_model(self):
        self.assertGreater(ReviewerModel.evaluate("cart","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")

    def test_reviewer_lda_model(self):
        self.assertGreater(ReviewerModel.evaluate("lda","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")

    def test_reviewer_svm_model(self):
        self.assertGreater(ReviewerModel.evaluate("svm","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")	

    def test_reviewer_nb_model(self):
        self.assertGreater(ReviewerModel.evaluate("nb","clothes&acc"), 0.9, "Accuracy must be superior to 90 percent")	
 
if __name__ == '__main__':
    unittest.main()
