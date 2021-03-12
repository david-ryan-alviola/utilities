import unittest
import model_utils as utils

from pydataset import data
from sklearn.model_selection import train_test_split

class TestModelUtils(unittest.TestCase):

    train, test = train_test_split(data("iris"), test_size=.2, random_state=1414)
    train, validate = train_test_split(train, test_size=.3, random_state=1414)

    def test_generate_xy_splits_positive(self):
        result = utils.generate_xy_splits(self.train, self.validate, self.test, target="Species")

        self.assertEqual(["X_train", "y_train", "X_validate", "y_validate", "X_test", "y_test"],
            list(result.keys()))
        self.assertFalse("Species" in result['X_train'])
        self.assertFalse("Species" in result['X_validate'])
        self.assertFalse("Species" in result['X_test'])
        self.assertIsNotNone(result['y_train'])
        self.assertIsNotNone(result['y_validate'])
        self.assertIsNotNone(result['y_test'])

    def test_generate_xy_splits_drop_columns_positive(self):
        result = utils.generate_xy_splits(self.train, self.validate, self.test, target="Species", drop_columns=["Sepal.Width", "Petal.Width"])

        self.assertFalse("Sepal.Width" in result['X_train'])
        self.assertFalse("Sepal.Width" in result['X_validate'])
        self.assertFalse("Sepal.Width" in result['X_test'])
        self.assertFalse("Petal.Width" in result['X_train'])
        self.assertFalse("Petal.Width" in result['X_validate'])
        self.assertFalse("Petal.Width" in result['X_test'])

if __name__ == "__main__":
    unittest.main()