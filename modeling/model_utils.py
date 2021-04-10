import pandas as pd

from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, \
    r2_score, explained_variance_score

def generate_xy_splits(train, validate, test, target, drop_columns=None):
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
    columns = []

    if (drop_columns != None):
        columns = drop_columns.copy()

    if target not in columns:
        columns.append(target)

    result['X_train'] = train.drop(columns=columns)
    result['y_train'] = pd.DataFrame(train[target])

    result['X_validate'] = validate.drop(columns=columns)
    result['y_validate'] = pd.DataFrame(validate[target])

    result['X_test'] = test.drop(columns=columns)
    result['y_test'] = pd.DataFrame(test[target])

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

def generate_baseline_model(prediction, key_name, target, y_train, y_validate):
    """
    Uses the provided baseline prediction value and appends it to a column on the train and validate \
    dataframes with the name of the provided key name. Evaluates the prediction's performance against \
    the target variable.
    """
    y_train[key_name] = prediction
    y_validate[key_name] = prediction

    rmse_train, rmse_validate = _calculate_rmse_values(target, key_name, y_train, y_validate)

    _print_rmse_comparison(rmse_train, rmse_validate, key_name)
    _print_train_and_validate_evals(y_train, y_validate, target, key_name)

def generate_regression_model(regressor, X_train, X_validate, y_train, y_validate, key_name, target):
    """
    Uses the provided regressor to predict the target variable and evaluates the model's performance.
    """
    regressor.fit(X_train, y_train[target])

    y_train[key_name] = regressor.predict(X_train)
    y_validate[key_name] = regressor.predict(X_validate)

    rmse_train, rmse_validate = _calculate_rmse_values(target, key_name, y_train, y_validate)

    _print_rmse_comparison(rmse_train, rmse_validate, key_name)
    _print_train_and_validate_evals(y_train, y_validate, target, key_name)

    return regressor

def apply_model_to_test_data(model, X_test, y_test, key_name, target):
    """
    Applies the provided model to the test data and evaluates its performance.
    """
    y_test[key_name] = model.predict(X_test)

    rmse_test = sqrt(mean_squared_error(y_test[target], y_test[key_name]))

    print(f"RMSE for {key_name} model\nOut-of-Sample Performance: {rmse_test}")
    _print_rsquare_and_variance(y_test, target, key_name)

def _calculate_rmse_values(target, key_name, y_train, y_validate):
    rmse_train = sqrt(mean_squared_error(y_train[target], y_train[key_name]))
    rmse_validate = sqrt(mean_squared_error(y_validate[target], y_validate[key_name]))

    return rmse_train, rmse_validate

def _print_rmse_comparison(rmse_train, rmse_validate, key_name):
    print(f"RMSE using {key_name}\nTrain/In-Sample: ", round(rmse_train, 4), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 4))

def _print_rsquare_and_variance(y_df, target, key_name):
    evs = explained_variance_score(y_df[target], y_df[key_name])
    r2 = r2_score(y_df[target], y_df[key_name])

    print(f"Explained variance:  {round(evs, 4)}")
    print(f"R-squared value:  {round(r2, 4)}")

def _print_train_and_validate_evals(y_train, y_validate, target, key_name):
    print("--------------------------------------------------")
    print("Train")
    _print_rsquare_and_variance(y_train, target, key_name)
    print("--------------------------------------------------")
    print("Validate")
    _print_rsquare_and_variance(y_validate, target, key_name)