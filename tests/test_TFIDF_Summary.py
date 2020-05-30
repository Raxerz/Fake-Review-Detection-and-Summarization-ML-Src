import sys
sys.path.append("../src")
import unittest
import TFIDFSummary
 
class TestTFIDSummary(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_summary_gen(self):
        self.assertEqual( TFIDFSummary.summaryGen("abc.txt","abc"), "Domain not in dataset")
 
if __name__ == '__main__':
    unittest.main()
