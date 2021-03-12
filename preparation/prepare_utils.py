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