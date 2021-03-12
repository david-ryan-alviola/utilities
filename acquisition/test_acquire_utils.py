import unittest
import acquire_utils as utils

from unittest.mock import patch
from pydataset import data

class TestAcquireUtils(unittest.TestCase):

    def test_generate_db_url(self):
        self.assertEqual(utils.generate_db_url("user", "password", "h.o.s.t", "db_name", "my_favorite_protocol"),
        "my_favorite_protocol://user:password@h.o.s.t/db_name")

    def test_generate_db_url_default_protocol(self):
        self.assertEqual(utils.generate_db_url("user", "password", "h.o.s.t", "db_name"),
        "mysql+pymysql://user:password@h.o.s.t/db_name")

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