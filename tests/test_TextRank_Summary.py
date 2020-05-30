import sys
sys.path.append("../src")
import unittest
import TextRank
 
class TextRankSummary(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_summary_gen(self):
        self.assertEqual( TextRank.summaryGen("abc.txt","abc"), "Domain not in dataset")
 
if __name__ == '__main__':
    unittest.main()
