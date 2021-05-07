import pandas as pd
import numpy as np
import unicodedata
import re
import json
import nltk

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.preprocessing import MinMaxScaler

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

    _print_positions(result['nan_positions'], "NaN values")
    _print_positions(result['empty_positions'], "Empty values")

    return result

def _print_positions(result, position_type):
    print(position_type)

    rows = pd.DataFrame(data=result[0], columns=['rows'])
    columns = pd.DataFrame(data=result[1], columns=['columns'])

    print(pd.concat([rows, columns], axis=1))
    print("--------------------------------")

def generate_outlier_bounds_iqr(df, column, multiplier=1.5):
    """
    Takes in a dataframe, the column name, and can specify a multiplier (default=1.5). Returns the upper and lower bounds for the
    values in that column that signify outliers.
    """
    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)
    iqr = q3 - q1

    upper = q3 + (multiplier * iqr)
    lower = q1 - (multiplier * iqr)

    return upper, lower

def generate_scaled_splits(train, validate, test, scaler=MinMaxScaler()):
    """
    Takes in a train, validate, test samples and can specify the type of scaler to use (default=MinMaxScaler). Returns the samples
    after scaling as dataframes.
    """
    scaler.fit(train)

    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns)
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

    return train_scaled, validate_scaled, test_scaled

def rfe(predictors, targets, model_type, k=1):
    """
    Takes in a a dataframe of predictors (ie X_train), the dataframe of targets (ie y_train), the type of model, and can specify 
    the amount of features you want (default k=1). Returns a list ordered by most important to least important feature. The top
    features will be assigned a rank of 1 (ie if k=2, there will be 2 features with a rank of 1).
    """
    model = model_type
    
    rfe = RFE(model, k)
    rfe.fit(predictors, targets)

    rfe_feature_mask = rfe.support_
    
    _print_ranks(rfe, predictors)

    return predictors.iloc[:, rfe_feature_mask].columns.tolist()

def select_kbest(predictors, targets, k=1):
    """
    Takes in a a dataframe of predictors (ie X_train), the dataframe of targets (ie y_train), and can specify 
    the amount of features you want (default k=1). Returns a list ordered by most important to least important feature.
    """
    f_selector = SelectKBest(f_regression, k=k)
    f_selector.fit(predictors, targets)

    feature_mask = f_selector.get_support()

    return predictors.iloc[:, feature_mask].columns.tolist()

def _print_ranks(selector, predictors):
    var_ranks = selector.ranking_
    var_names = predictors.columns.tolist()

    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    print(rfe_ranks_df.sort_values('Rank'))

def show_missing_value_stats_by_col(df):
    """
    Takes in a data frame and returns information on missing values in each column.
    """
    cols = df.columns
    rows = len(df)
    result = pd.DataFrame(index=cols, columns=['num_rows_missing', 'pct_rows_missing'])
    pd.set_option('max_rows', rows)
    
    result['num_rows_missing'] = df.isnull().sum()
    result['pct_rows_missing'] = round(df.isnull().sum() / rows, 6)
    
    return result

def show_missing_value_stats_by_row(df):
    """
    Takes in a data frame and returns information on missing values in each row.
    """
    total_cols = df.shape[1]
    total_rows = df.shape[0]
    result = pd.DataFrame(df.isnull().sum(axis=1).value_counts(), columns=['num_rows'])
    pd.set_option('max_rows', total_rows)
    
    result = result.reset_index()
    result = result.rename(columns={'index' : 'num_cols_missing'})
    result['pct_cols_missing'] = result['num_cols_missing'] / total_cols
    result = result.set_index('num_cols_missing')
    result = result.sort_values('num_cols_missing', ascending=True)
    
    return result

def handle_missing_values(df, col_thresh, row_thresh):
    """
    Takes in a data frame and thresholds (0.0 - 1.0) for columns and rows and returns the data frame after dropping the rows and
    columns that are not populated at the specified threshold amounts.
    """
    req_col = int(round(col_thresh * len(df.index), 0))
    req_row = int(round(row_thresh * len(df.columns), 0))
    
    df = df.dropna(axis=1, thresh=req_col)
    df = df.dropna(axis=0, thresh=req_row)
    
    return df

def set_index_to_datetime(df, column_name):
    """
    Takes in a dataframe and the column name of a column containing string values formatted as dates. This function converts the
    column to a pandas.Datetime object and sets as the index of the dataframe then returns the dataframe sorted by index.
    """
    date_df = df.copy()

    date_df[column_name] = pd.to_datetime(date_df[column_name])

    return date_df.set_index(column_name).sort_index()

def find_outliers_with_sigma(df, cols, sigma):
    """
    Generates a dictionary containing the lower and upper bounds indicating outliers for the given variables determined with the sigma decision rule.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the variables
    cols : list of strings
        List containing the key names of the variables to calculate outlier bounds for

    Returns
    -------
    dict
        Dictionary containing the outlier bounds where each key is the column name of the variable.
    """

    outliers_dict = {}
    
    for col in cols:
        x = df[col]

        zscores = pd.Series((x - x.mean()) / x.std())
        outliers = x[zscores.abs() >= sigma]
        
        outliers_dict[col + "_outliers"] = outliers
    
    return outliers_dict

def nlp_basic_clean(string):
    """
    Lowercases, removes non-ASCII characters, and removes non-alphanumeric (except ' or \s') from the passed in string.
    """
    
    cleaned_string = string
    
    cleaned_string = cleaned_string.lower()
    cleaned_string = unicodedata.normalize("NFKD", cleaned_string).encode("ascii", "ignore").decode("utf-8", "ignore")
    cleaned_string = re.sub(r"[^a-z0-9'\s]", "", cleaned_string)
    
    return cleaned_string

def nlp_tokenize(string):
    """
    Applies the ToktokTokenizer.tokenize() method to the passed in string.
    """
    
    tokenized_string = string
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    return tokenizer.tokenize(tokenized_string, return_str=True)

def nlp_stem(string):
    """
    Generates a list of the stem for each word in the original string and returns the joined list of stems as a single string.
    """
    
    ps = nltk.porter.PorterStemmer()
    
    stems = [ps.stem(word) for word in string.split()]
    
    return " ".join(stems)

def nlp_lemmatize(string):
    """
    Generates a list of the root word for each word in the original string and returns the joined list of root words as a single string.
    """
    
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    return " ".join(lemmas)

def nlp_remove_stopwords(string, extra_words=None, exclude_words=None):
    """
    Generates a list of stop words then adds and/or removes the specified words from that stop word list. Any words from the \
    original string that are not present in the stop word list are added to the filtered word list. Returns the joined filtered word \
    list as a single string.
    """
    
    stopword_list = stopwords.words("english")
    extras = []
    excludes = []
    
    if extra_words != None:
        extras = extra_words.copy()
        
    if exclude_words != None:
        excludes = exclude_words.copy()
    
    for extra_word in extras:
        if extra_word not in stopword_list:
            stopword_list.append(extra_word)
        
    for exclude_word in excludes:
        if exclude_word in stopword_list:
            stopword_list.remove(exclude_word)

    filtered_words = [word for word in string.split() if word not in stopword_list]
    
    return " ".join(filtered_words)