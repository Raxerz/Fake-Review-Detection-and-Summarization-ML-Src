import sys
sys.path.append("../src")
import unittest
import SummaryMain
 
class TestSummaryMain(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_summary_gen(self):
        self.assertEqual( SummaryMain.main(5,[]), "Wrong choice")
 
if __name__ == '__main__':
    unittest.main()
