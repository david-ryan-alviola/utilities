import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from acquire.acquire_utils import generate_csv_url, generate_db_url, generate_df
from prepare.prepare_utils import nan_null_empty_check, split_dataframe
from explore.explore_utils import explore_univariate, explore_bivariate, explore_multivariate
from explore.stats_utils import evaluate_hypothesis_pcorrelation, evaluate_hypothesis_ttest

def generate_xy_splits(train, validate, test, target, drop_columns=[]):
    """
    Utility function that splits samples into X and y values.

    This function removes the target variable and any other columns from the dataframe to prepare for model fitting.

    Parameters
    ----------
    train : DataFrame
        The training sample.
    validate : DataFrame
        The validate sample.
    test : DataFrame
        The test sample.
    target : str
        The target variable name.
    drop_columns : list of str
        List containing the names of columns to drop.

    Returns
    -------
    dict
        Dictionary containing the different splits with keys:  'X_train', 'y_train', 'X_validate', 'y_validate', 'X_test', 'y_test'
    """

    result = {}

    if target not in drop_columns:
        drop_columns.append(target)

    result['X_train'] = train.drop(columns=drop_columns)
    result['y_train'] = train[target]

    result['X_validate'] = validate.drop(columns=drop_columns)
    result['y_validate'] = validate[target]

    result['X_test'] = test.drop(columns=drop_columns)
    result['y_test'] = test[target]

    return result

def get_metrics_bin(clf, X, y):
    '''
    get_metrics_bin will take in a sklearn classifier model, an X and a y variable and utilize
    the model to make a prediction and then gather accuracy, class report evaluations
    Credit to @madeleine-capper

    return:  a classification report as a pandas DataFrame
    '''
    y_pred = clf.predict(X)
    accuracy = clf.score(X, y)
    conf = confusion_matrix(y, y_pred)
    class_report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).T
    tpr = conf[1][1] / conf[1].sum()
    fpr = conf[0][1] / conf[0].sum()
    tnr = conf[0][0] / conf[0].sum()
    fnr = conf[1][0] / conf[1].sum()
    print(f'''
    The accuracy for our model is {accuracy:.4}
    The True Positive Rate is {tpr:.3}, The False Positive Rate is {fpr:.3},
    The True Negative Rate is {tnr:.3}, and the False Negative Rate is {fnr:.3}
    ''')
    return class_report