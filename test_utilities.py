import unittest
import utilities as utils

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

if __name__ == "__main__":
    unittest.main()