import unittest
import utilities as utils

class TestUtilityFunctions(unittest.TestCase):

    def test_generate_db_url(self):
        self.assertEqual(utils.generate_db_url("user", "password", "h.o.s.t", "db_name", "my_favorite_protocol"),
        "my_favorite_protocol://user:password@h.o.s.t/db_name")

    def test_generate_db_url_default_protocol(self):
        self.assertEqual(utils.generate_db_url("user", "password", "h.o.s.t", "db_name"),
        "mysql+pymysql://user:password@h.o.s.t/db_name")

    def test_evaluate_hypothesis_ttest_two_tailed_reject_null(self):
        self.assertTrue(utils.evaluate_hypothesis_ttest(.01, 14))

    def test_evaluate_hypothesis_ttest_two_tailed_fail_to_reject_null(self):
        self.assertFalse(utils.evaluate_hypothesis_ttest(4, 0))

    def test_evaluate_hypothesis_ttest_greater_reject_null(self):
        self.assertTrue(utils.evaluate_hypothesis_ttest(.01, 14, tails = "greater"))

    def test_evaluate_hypothesis_ttest_greater_fail_to_reject_null(self):
        self.assertFalse(utils.evaluate_hypothesis_ttest(.01, -14, tails = "greater"))

    def test_evaluate_hypothesis_ttest_less_reject_null(self):
        self.assertTrue(utils.evaluate_hypothesis_ttest(.01, -14, tails = "less"))

    def test_evaluate_hypothesis_ttest_less_fail_to_reject_null(self):
        self.assertFalse(utils.evaluate_hypothesis_ttest(.01, 14, tails = "less"))

    def test_evaluate_hypothesis_ttest_tails_negative(self):
        with self.assertRaises(ValueError):
            utils.evaluate_hypothesis_ttest(.01, 14, tails = "Sonic")


if __name__ == "__main__":
    unittest.main()