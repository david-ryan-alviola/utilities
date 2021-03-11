import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from explore import explore_univariate, explore_bivariate, explore_multivariate

def evaluate_hypothesis_ttest(p_value, t_value, alpha = .05, tails = "two", null_hypothesis = "", alternative_hypothesis = ""):
    """
    Utility function to evaluate T-test hypothesis

    Evaluates whether or not the null hypothesis can be rejected after performing T-test analysis.

    Parameters
    ----------

    p_value : float
        The p_value from the T-test
    t_value : float
        The t_value from the T-test
    alpha : float, optional
        The specified alpha value or .05 (default) if unspecified
    tails : {'two', 'greater', 'less'}, optional
        Defines how to evaluate the hypothesis based on T-test:
            * 'two' : evaluates hypothesis with criteria for two-tailed T-test (default)
            * 'greater' : evaluates hypothesis with criteria for one-tailed T-test
               when examining if an increase in the value is present
            * 'less' : evaluates hypothesis with criteria for one-tailed T-test
               when examining if a decrease in the value is present
    null_hypothesis : str, optional
        The null hypothesis being tested. Empty string by default.
    alternative_hypothesis : str, optional
        The alternative hypothesis being tested. Empty string by default.
    
    Returns
    -------
    dict
        Contains boolean value telling if the null hypothesis can be rejected and message about the result.
    """
    def fail_to_reject_null_hypothesis():
        return f"We fail to reject the null hypothesis:  {null_hypothesis}"

    def reject_null_hypothesis():
        return f"We reject the null hypothesis. We move forward with the alternative hypothesis:  {alternative_hypothesis}"

    result = {'t' :  t_value, 'p' :  p_value, 'a' :  alpha}

    print("------------------------------------------")
    print(f"t:  {t_value}, p:  {p_value}, a:  {alpha}")
    print()

    if tails == "two":

        if p_value < alpha:
            result['message'] = reject_null_hypothesis()
            result['reject_null'] = True

        else:
            result['message'] = fail_to_reject_null_hypothesis()
            result['reject_null'] = False

    else:

        if (p_value / 2) < alpha:

            if (tails == "greater"):
                if (t_value > 0):
                    result['message'] = reject_null_hypothesis()
                    result['reject_null'] = True

                else:
                    result['message'] = fail_to_reject_null_hypothesis()
                    result['reject_null'] = False

            elif (tails == "less"):
                if (t_value < 0):
                    result['message'] = reject_null_hypothesis()
                    result['reject_null'] = True
                
                else:
                    result['message'] = fail_to_reject_null_hypothesis()
                    result['reject_null'] = False

            else:
                raise ValueError("tails parameter only accepts:  'two', 'greater', 'less'")

        else:
            result['message'] = fail_to_reject_null_hypothesis()
            result['reject_null'] = False

    print(result['message'])
    print("------------------------------------------")

    return result

def generate_db_url(user, password, host, db_name, protocol = "mysql+pymysql"):
    """
    Utility function for generating database URL

    This function generates a database URL with the mysql and pymysql protocol for use with pandas.read_sql()

    Parameters
    ----------
    user : str
        The username for the database
    password : str
        The password for the database
    host : str
        The host address for the database server
    db_name : str
        The name of the database
    protocol : str
        The protocol to be used. Defaults to "mysql+pymysql".
    
    Returns
    -------
    str
        URL in this format:  protocol://user:password@host/db_name
    """
    return f"{protocol}://{user}:{password}@{host}/{db_name}"

def evaluate_hypothesis_pcorrelation(correlation, p_value, alpha = .05, null_hypothesis = "", alternative_hypothesis = ""):
    """
    Utility function to evaluate T-test hypothesis

    Evaluates whether or not the null hypothesis can be rejected after performing T-test analysis.

    Parameters
    ----------
    correlation : float
        The correlation value from the calculation
    p_value : float
        The p_value from the calculation
    alpha : float, optional
        The specified alpha value or .05 (default) if unspecified
    null_hypothesis : str, optional
        The null hypothesis being tested. Empty string by default.
    alternative_hypothesis : str, optional
        The alternative hypothesis being tested. Empty string by default.
    
    Returns
    -------
    dict
        Contains boolean value telling if the null hypothesis can be rejected and message about the result.
    """

    def fail_to_reject_null_hypothesis():
        return f"We fail to reject the null hypothesis:  {null_hypothesis}"

    def reject_null_hypothesis():
        return f"We reject the null hypothesis. We move forward with the alternative hypothesis:  {alternative_hypothesis}"

    print("------------------------------------------")
    print(f"corr:  {correlation}, p:  {p_value}, a:  {alpha}")
    print()

    result = {'corr' :  correlation, 'p' :  p_value, 'a' :  alpha}

    if correlation > 1 or correlation < -1:
        raise ValueError("The correlation must be between -1 and 1")
    else:
        if p_value < alpha:
            result['reject_null'] = True
            result['message'] = reject_null_hypothesis()
        else:
            result['reject_null'] = False
            result['message'] = fail_to_reject_null_hypothesis()

        if correlation > 0:
            result['correlation'] = "positive"
        
        elif correlation < 0:
            result['correlation'] = "negative"
        
        else:
            result['correlation'] = "none"

        print(result['message'])
        print(f"Correlation direction:  {result['correlation']}")
        print("------------------------------------------")

        return result

def generate_csv_url(sheet_url):
    """
    Utility function for generating csv URL from a google sheets link

    This function generates a link to a csv file from a link used to edit a google sheets file.
    The gid must be present in the URL.

    Parameters
    ----------
    sheet_url : str
        The URL for the google sheet file
    
    Returns
    -------
    str
        URL for the csv file
    """
    if type(sheet_url) == str:

        if(sheet_url.find("edit#gid") > -1):
            return sheet_url.replace("edit#gid", "export?format=csv&gid")

        else:
            raise ValueError("sheet_url must contain 'edit#gid' phrase")
    else:
        raise TypeError("sheet_url must be a string")

def generate_df(file_name, query="", db_url="", cached=True):
    """
    Utilty function for generating a dataframe

    This function generates a dataframe either from an existing CSV file or from the provided query and database connection.
    If the CSV with the file_name is present, the dataframe will be generated from the CSV file, otherwise the dataframe
    will be generated from the database result and a CSV file will be created with the provided file_name. This behavior
    can be overridden if cached is set to False.

    Parameters
    ----------
    file_name : str
        The name of the file
    query : str, optional
        The query for the data. Empty string by default.
    db_url : str, optional
        URL for database connection. Empty string by default.
    cached : bool, optional
        Indicates whether or not dataframe should be generated from existing CSV file. Default is True.

    Returns
    -------
    DataFrame
        Dataframe containing the data from CSV or query
    """
    file_present = os.path.isfile(file_name)

    if cached and file_present:
        df = pd.read_csv(file_name)
    else:
        df = pd.read_sql(query, db_url)
        df.to_csv(file_name)

    return df

def split_dataframe(df, stratify_by=None, rand=1414, test_size=.2, validate_size=.3):
    """
    Utility function to create train, validate, and test splits.

    Generates train, validate, and test samples from a dataframe.
    Credit to @ryanorsinger

    Parameters
    ----------
    df : DataFrame
        The dataframe to be split
    stratify_by : str
        Name of the target variable. Ensures different results of target variable are spread between the samples. Default is None.
    test_size : float
        Ratio of dataframe (0.0 to 1.0) that should be kept for testing sample. Default is 0.2.
    validate_size: float
        Ratio of train sample (0.0 to 1.0) that should be kept for validate sample. Default is 0.3.
    random_stat : int
        Value provided to create re-produceable results. Default is 1414.

    Returns
    -------
    DataFrame
        Three dataframes representing the training, validate, and test samples
    """
    
    if stratify_by == None:
        train, test = train_test_split(df, test_size=test_size, random_state=rand)
        train, validate = train_test_split(train, test_size=validate_size, random_state=rand)
    else:
        train, test = train_test_split(df, test_size=test_size, random_state=rand, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=validate_size, random_state=rand, stratify=train[stratify_by])

    return train, validate, test

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

def nan_null_empty_check(df):
    """
    Utility function that checks for missing values in a dataframe.

    This function will return a tuple containing the positions of NaN, None, NaT, or empty strings.

    Parameters
    ----------
    df : DataFrame
        The dataframe that you want to search.
    
    Returns
    -------
    tuple
        A tuple containing coordinates of the missing values:  ([rows], [columns])
    """
    result = {}
    
    result['nan_positions'] = np.where(pd.isna(df))
    result['empty_positions'] = np.where(df.applymap(lambda x: str(x).strip() == ""))

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

def explore_one_dimension(train, cat_vars, quant_vars):
    """
    Wrapper for @magsguist explore_univariate function
    """
    explore_univariate(train, cat_vars, quant_vars)

def explore_two_dimensions(train, target, cat_vars, quant_vars):
    """
    Wrapper for @magsguist explore_bivariate function
    """
    explore_bivariate(train, target, cat_vars, quant_vars)

def explore_three_dimensions(train, target, cat_vars, quant_vars):
    """
    Wrapper for @magsguist explore_multivariate function
    """
    explore_multivariate(train, target, cat_vars, quant_vars)