import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_column_counts_df(df, target_col):
    """
    Generates a dataframe containing the counts of a target variable within the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable and the counts of the target occurring within the dataframe.
    """

    return df[target_col].value_counts(dropna=False).reset_index().\
                rename(columns={'index': target_col, target_col : target_col + '_count'})

def generate_column_probability_df(df, target_col):
    """
    Generates a dataframe containing the probability of a target variable within the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable and the probabilities of the target occurring within the dataframe.
    """

    return (df[target_col].value_counts(dropna=False)/df[target_col].count()).reset_index().\
                rename(columns={'index': target_col, target_col : target_col + '_proba'})

def generate_counts_and_probability_df(df, target_var):
    """
    Generates a dataframe containing the counts and probabilities of a target variable within the dataframe.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable, the counts, and probabilities of the target occurring within the dataframe.
    """

    counts_df = generate_column_counts_df(df, target_var)
    probability_df = generate_column_probability_df(df, target_var)

    return counts_df.merge(probability_df)

def visualize_target_counts(df, target_var, target_counts, fig_size=(12, 8)):
    """
    Creates a barplot of the different values for the target variable and the count for each.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target variable
    target_var : string
        The key name for the target variable column
    target_counts : string
        The key name for the counts of the target variable column
    fig_size : tuple of (int, int)
        The dimensions for the barplot (default=(12,8))

    Returns
    -------
    None
    """

    plt.figure(figsize=fig_size)
    
    splot = sns.barplot(data=df, x = target_var, y = target_counts, ci = None)

    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', xytext = (0, 10), 
                       textcoords = 'offset points'
                       )
    plt.xticks(rotation='vertical')
    plt.show()

def generate_conditional_probability_df(df, target_var, conditional_var):
    """
    Generates a dataframe containing the conditional probability of a target variable occurring given a conditional variable's presence.

    Parameters
    ----------
    df : DataFrame
        The dataframe containing the target and conditional variables
    target_var : string
        The key name for the target variable column
    conditional_var : string
        The key name for the conditional variable column

    Returns
    -------
    DataFrame
        Dataframe containing the target variable, the conditional variable, and the probability of the combination occurring within the dataframe.
    """

    probabilities = df.groupby([conditional_var]).size().div(len(df))

    conditional_proba_df = pd.DataFrame(df.groupby([conditional_var, target_var]).size().div(len(df)).\
                              div(probabilities, axis=0, level=conditional_var).\
                              reset_index().\
                              rename(columns={0 : 'proba_' + target_var + '_given_' + conditional_var}))

    return conditional_proba_df