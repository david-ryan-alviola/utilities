import unittest
import utilities as utils

from unittest.mock import patch
from pydataset import data

class TestUtilityFunctions(unittest.TestCase):

    reject_null_message = "We reject the null hypothesis. We move forward with the alternative hypothesis:  "
    fail_to_reject_null_message = "We fail to reject the null hypothesis:  "

    def test_generate_db_url(self):
        self.assertEqual(utils.generate_db_url("user", "password", "h.o.s.t", "db_name", "my_favorite_protocol"),
        "my_favorite_protocol://user:password@h.o.s.t/db_name")

    def test_generate_db_url_default_protocol(self):
        self.assertEqual(utils.generate_db_url("user", "password", "h.o.s.t", "db_name"),
        "mysql+pymysql://user:password@h.o.s.t/db_name")

    def test_evaluate_hypothesis_ttest_two_tailed_reject_null(self):
        result = utils.evaluate_hypothesis_ttest(.01, 14)

        self.assertTrue(result['reject_null'])
        self.assertEqual(result['message'], self.reject_null_message)

    def test_evaluate_hypothesis_ttest_two_tailed_fail_to_reject_null(self):
        result = utils.evaluate_hypothesis_ttest(4, 0)
        
        self.assertFalse(result['reject_null'])
        self.assertEqual(result['message'], self.fail_to_reject_null_message)

    def test_evaluate_hypothesis_ttest_greater_reject_null(self):
        result = utils.evaluate_hypothesis_ttest(.01, 14, tails = "greater")
        
        self.assertTrue(result['reject_null'])
        self.assertEqual(result['message'], self.reject_null_message)

    def test_evaluate_hypothesis_ttest_greater_fail_to_reject_null(self):
        result = utils.evaluate_hypothesis_ttest(.01, -14, tails = "greater")
        
        self.assertFalse(result['reject_null'])
        self.assertEqual(result['message'], self.fail_to_reject_null_message)

    def test_evaluate_hypothesis_ttest_less_reject_null(self):
        result = utils.evaluate_hypothesis_ttest(.01, -14, tails = "less")
        
        self.assertTrue(result['reject_null'])
        self.assertEqual(result['message'], self.reject_null_message)

    def test_evaluate_hypothesis_ttest_less_fail_to_reject_null(self):
        result = utils.evaluate_hypothesis_ttest(.01, 14, tails = "less")
        
        self.assertFalse(result['reject_null'])
        self.assertEqual(result['message'], self.fail_to_reject_null_message)

    def test_evaluate_hypothesis_ttest_tails_negative(self):
        with self.assertRaises(ValueError):
            utils.evaluate_hypothesis_ttest(.01, 14, tails = "Sonic")
    
    def test_evaluate_hypothesis_pcorrelation_reject_null_positive_corr(self):
        result = utils.evaluate_hypothesis_pcorrelation(1, .04)

        self.assertTrue(result['reject_null'])
        self.assertEqual(result['message'], self.reject_null_message)
        self.assertEqual(result['correlation'], "positive")
        # used to test bug where alternative hypothesis was being printed instead of alpha value
        self.assertEqual(0.05, result['a'])

    def test_evalutate_hypothesis_pcorrelation_fail_to_reject_null_positive_corr(self):
        result = utils.evaluate_hypothesis_pcorrelation(1, .06)

        self.assertFalse(result['reject_null'])
        self.assertEqual(result['message'], self.fail_to_reject_null_message)
        self.assertEqual(result['correlation'], "positive")

    def test_evaluate_hypothesis_pcorrelation_reject_null_negative_corr(self):
        result = utils.evaluate_hypothesis_pcorrelation(-1, .04)

        self.assertTrue(result['reject_null'])
        self.assertEqual(result['message'], self.reject_null_message)
        self.assertEqual(result['correlation'], "negative")

    def test_evaluate_hypothesis_pcorrelation_fail_to_reject_null_negative_corr(self):
        result = utils.evaluate_hypothesis_pcorrelation(-1, .06)

        self.assertFalse(result['reject_null'])
        self.assertEqual(result['message'], self.fail_to_reject_null_message)
        self.assertEqual(result['correlation'], "negative")

    def test_evaluate_hypothesis_pcorrelation_reject_null_no_corr(self):
        result = utils.evaluate_hypothesis_pcorrelation(0, .04)

        self.assertTrue(result['reject_null'])
        self.assertEqual(result['message'], self.reject_null_message)
        self.assertEqual(result['correlation'], "none")

    def test_evaluate_hypothesis_pcorrelation_fail_to_reject_null_no_corr(self):
        result = utils.evaluate_hypothesis_pcorrelation(0, .06)

        self.assertFalse(result['reject_null'])
        self.assertEqual(result['message'], self.fail_to_reject_null_message)
        self.assertEqual(result['correlation'], "none")

    def test_evaluate_hypothesis_pcorrelation_negative(self):
        with self.assertRaises(ValueError):
            utils.evaluate_hypothesis_pcorrelation(-14, .06)
        
        with self.assertRaises(ValueError):
            utils.evaluate_hypothesis_pcorrelation(14, .06)

    def test_generate_csv_url_positive(self):
        self.assertEqual(
            utils.generate_csv_url("https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=1234"), 
            "https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/export?format=csv&gid=1234")

    def test_generate_csv_url_negative(self):
        with self.assertRaises(TypeError):
            utils.generate_csv_url(14398)

        with self.assertRaises(ValueError):
            utils.generate_csv_url("www.google.com")

    @patch("pandas.read_sql", return_value=data("iris"))
    @patch("pandas.DataFrame.to_csv", return_value="file.csv")
    @patch("os.path.isfile", return_value=True)
    def test_generate_df_not_cached_positive(self, mock_read_sql, mock_to_csv, mock_isfile):
        self.assertEqual(data("iris").Species.all(), 
            utils.generate_df("file.csv", "query", "db_url", cached=False).Species.all())

    @patch("pandas.read_csv", return_value=data("iris"))
    @patch("os.path.isfile", return_value=True)
    def test_generate_df_cached_positive(self, mock_read_csv, mock_isfile):
        self.assertEqual(data("iris").Species.all(),
            utils.generate_df("file.csv").Species.all())

if __name__ == "__main__":
    unittest.main()