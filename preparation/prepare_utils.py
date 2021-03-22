import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

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

def split_dataframe_continuous_target(dframe, target, bins=5, rand=1414, test_size=.2, validate_size=.3):
    """
    Utility function to create train, validate, and test splits when targeting a continuous variable.

    Generates train, validate, and test samples from a dataframe when targeting a continuous variable.
    Credit to @ryanorsinger

    Parameters
    ----------
    df : DataFrame
        The dataframe to be split
    target : str
        Name of the continuous target variable. Ensures different results of target variable are spread between the samples.
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
    df = dframe.copy()
    binned_y = pd.cut(df[target], bins=bins, labels=list(range(bins)))
    df["bins"] = binned_y

    train_validate, test = train_test_split(df, stratify=df["bins"], test_size=test_size, random_state=rand)
    train, validate = train_test_split(train_validate, stratify=train_validate["bins"], test_size=validate_size, random_state=rand)

    train = train.drop(columns=["bins"])
    validate = validate.drop(columns=["bins"])
    test = test.drop(columns=["bins"])
    
    return train, test, validate

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

    _print_positions(result['nan_positions'], "NaN values")
    _print_positions(result['empty_positions'], "Empty values")

    return result

def _print_positions(result, position_type):
    print(position_type)

    rows = pd.DataFrame(data=result[0], columns=['rows'])
    columns = pd.DataFrame(data=result[1], columns=['columns'])

    print(pd.concat([rows, columns], axis=1))
    print("--------------------------------")
