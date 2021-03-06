import sys
sys.path.append("../src")
import unittest
import ReviewModel
 
class TestReviewModel(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_review_model_path(self):
        self.assertEqual( ReviewModel.evaluate("abc","def"), "One or more files do not exist")

    def test_review_knn_model(self):
        self.assertGreater(ReviewModel.evaluate("knn","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")	

    def test_review_cart_model(self):
        self.assertGreater(ReviewModel.evaluate("cart","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")

    def test_review_lda_model(self):
        self.assertGreater(ReviewModel.evaluate("lda","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")

    def test_review_svm_model(self):
        self.assertGreater(ReviewModel.evaluate("svm","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")	

    def test_review_nb_model(self):
        self.assertGreater(ReviewModel.evaluate("nb","musical-instruments"), 0.9, "Accuracy must be superior to 90 percent")	
 
if __name__ == '__main__':
    unittest.main()
