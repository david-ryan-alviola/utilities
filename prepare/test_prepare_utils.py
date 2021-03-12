import unittest
import numpy as np
import pandas as pd
import prepare_utils as utils

from pydataset import data

class TestPrepareUtils(unittest.TestCase):

    test_df = pd.DataFrame(columns=['A', 'B', 'C', 'D'], data=[["value", np.nan, None, "   "]])

    def test_nan_null_empty_check(self):
        expected = {'nan_positions' : (np.array([0, 0]), np.array([1, 2])), 'empty_positions' : (np.array([0]), np.array([3]))}
        result = utils.nan_null_empty_check(self.test_df)
        
        self.assertEqual(expected['nan_positions'][0].all(), result['nan_positions'][0].all())
        self.assertEqual(expected['nan_positions'][1].all(), result['nan_positions'][1].all())
        self.assertEqual(expected['empty_positions'][0].all(), result['empty_positions'][0].all())
        self.assertEqual(expected['empty_positions'][1].all(), result['empty_positions'][1].all())

if __name__ == "__main__":
    unittest.main()