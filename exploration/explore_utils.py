import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats

# Credit to @magsguist
def explore_univariate(train, cat_vars, quant_vars):
    for var in cat_vars:
        _explore_univariate_categorical(train, var)
        print('_________________________________________________________________')
    for col in quant_vars:
        p, descriptive_stats = _explore_univariate_quant(train, col)
        plt.show(p)
        print(descriptive_stats)

def explore_bivariate_categorical(target, cat_vars, train):
    """
    Takes in a target and plots it against categorial variables. Outputs boxplots and barplots and gives the mean of the target
    by each categorical variable.
    """
    for var in cat_vars:
        _print_header(var, target)

        sns.boxplot(x=var, y=target, data=train)
        plt.show()

        print()

        sns.barplot(x=var, y=target, data=train)
        plt.show()
        
        print("-------------------------------")
        print(f"Mean {target} by {var}:  ")
        print(train.groupby(var)[target].mean())
        print()

def explore_bivariate_continuous(target, cont_vars, train):
    """
    Takes in a target and plots it against continuous variables. Outputs a relplot and calculates the corrleation value between
    the target and each continuous variable.
    """
    for var in cont_vars:
        _print_header(var, target)
        
        sns.relplot(x=var, y=target, data=train)
        plt.show()
        corr, p = stats.pearsonr(train[var], train[target])
        
        print("-------------------------------")
        print(f"Correlation between {var} and {target}:  {corr}")
        print(f"P value:  {p}")
        print()

def explore_multivariate_(cont_vars, cat_vars, target, train):
    """
    Takes in a target and continuous and categorical variables. Outputs a relplot of each continuous variable against the target
    with each categorical varible as the hue.
    """
    for cont_var in cont_vars:
        _print_header(cont_var, target)
        
        for cat_var in cat_vars:
            sns.relplot(x=cont_var, y=target, hue=cat_var, data=train)
            plt.title(f"By {cat_var}")
            plt.show()
            print()

def _print_header(var, target):
    print(f"{var} vs {target}")
    print("-------------------------------")

@DeprecationWarning
def explore_bivariate(train, target, cat_vars, quant_vars):
    for cat in cat_vars:
        _explore_bivariate_categorical(train, target, cat)
    for quant in quant_vars:
        _explore_bivariate_quant(train, target, quant)

@DeprecationWarning
def explore_multivariate(train, target, cat_vars, quant_vars):
    '''
    '''
    _plot_swarm_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    _plot_violin_grid_with_color(train, target, cat_vars, quant_vars)
    plt.show()
    sns.pairplot(data=train, vars=quant_vars, hue=target)
    plt.show()
    _plot_all_continuous_vars(train, target, quant_vars)
    plt.show()    


### Univariate

def _explore_univariate_categorical(train, cat_var):
    '''
    takes in a dataframe and a categorical variable and returns
    a frequency table and barplot of the frequencies. 
    '''
    frequency_table = _freq_table(train, cat_var)
    plt.figure(figsize=(2,2))
    sns.barplot(x=cat_var, y='Count', data=frequency_table, color='lightseagreen')
    plt.title(cat_var)
    plt.show()
    print(frequency_table)

def _explore_univariate_quant(train, quant_var):
    '''
    takes in a dataframe and a quantitative variable and returns
    descriptive stats table, histogram, and boxplot of the distributions. 
    '''
    descriptive_stats = train[quant_var].describe()
    plt.figure(figsize=(8,2))

    p = plt.subplot(1, 2, 1)
    p = plt.hist(train[quant_var], color='lightseagreen')
    p = plt.title(quant_var)

    # second plot: box plot
    p = plt.subplot(1, 2, 2)
    p = plt.boxplot(train[quant_var])
    p = plt.title(quant_var)
    return p, descriptive_stats
    
def _freq_table(train, cat_var):
    '''
    for a given categorical variable, compute the frequency count and percent split
    and return a dataframe of those values along with the different classes. 
    '''
    class_labels = list(train[cat_var].unique())

    frequency_table = (
        pd.DataFrame({cat_var: class_labels,
                      'Count': train[cat_var].value_counts(normalize=False), 
                      'Percent': round(train[cat_var].value_counts(normalize=True)*100,2)}
                    )
    )
    return frequency_table


#### Bivariate

def _explore_bivariate_categorical(train, target, cat_var):
    '''
    takes in categorical variable and binary target variable, 
    returns a crosstab of frequencies
    runs a chi-square test for the proportions
    and creates a barplot, adding a horizontal line of the overall rate of the target. 
    '''
    print(cat_var, "\n_____________________\n")
    ct = pd.crosstab(train[cat_var], train[target], margins=True)
    chi2_summary, observed, expected = _run_chi2(train, cat_var, target)
    p = _plot_cat_by_target(train, target, cat_var)
    
    print(chi2_summary)
    print("\nobserved:\n", ct)
    print("\nexpected:\n", expected)
    plt.show(p)
    print("\n_____________________\n")

def _explore_bivariate_quant(train, target, quant_var):
    '''
    descriptive stats by each target class. 
    compare means across 2 target groups 
    boxenplot of target x quant
    swarmplot of target x quant
    '''
    print(quant_var, "\n____________________\n")
    descriptive_stats = train.groupby(target)[quant_var].describe()
    average = train[quant_var].mean()
    mann_whitney = _compare_means(train, target, quant_var)
    plt.figure(figsize=(4,4))
    boxen = _plot_boxen(train, target, quant_var)
    swarm = _plot_swarm(train, target, quant_var)
    plt.show()
    print(descriptive_stats, "\n")
    print("\nMann-Whitney Test:\n", mann_whitney)
    print("\n____________________\n")

## Bivariate Categorical

def _run_chi2(train, cat_var, target):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    chi2_summary = pd.DataFrame({'chi2': [chi2], 'p-value': [p], 
                                 'degrees of freedom': [degf]})
    expected = pd.DataFrame(expected)
    return chi2_summary, observed, expected

def _plot_cat_by_target(train, target, cat_var):
    p = plt.figure(figsize=(2,2))
    p = sns.barplot(cat_var, target, data=train, alpha=.8, color='lightseagreen')
    overall_rate = train[target].mean()
    p = plt.axhline(overall_rate, ls='--', color='gray')
    return p
    

## Bivariate Quant

def _plot_swarm(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.swarmplot(data=train, x=target, y=quant_var, color='lightgray')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

def _plot_boxen(train, target, quant_var):
    average = train[quant_var].mean()
    p = sns.boxenplot(data=train, x=target, y=quant_var, color='lightseagreen')
    p = plt.title(quant_var)
    p = plt.axhline(average, ls='--', color='black')
    return p

# alt_hyp = ‘two-sided’, ‘less’, ‘greater’

def _compare_means(train, target, quant_var, alt_hyp='two-sided'):
    x = train[train[target]==0][quant_var]
    y = train[train[target]==1][quant_var]
    return stats.mannwhitneyu(x, y, use_continuity=True, alternative=alt_hyp)


### Multivariate

def _plot_all_continuous_vars(train, target, quant_vars):
    '''
    Melt the dataset to "long-form" representation
    boxenplot of measurement x value with color representing survived. 
    '''
    my_vars = [item for sublist in [quant_vars, [target]] for item in sublist]
    sns.set(style="whitegrid", palette="muted")
    melt = train[my_vars].melt(id_vars=target, var_name="measurement")
    plt.figure(figsize=(8,6))
    p = sns.boxenplot(x="measurement", y="value", hue=target, data=melt)
    p.set(yscale="log", xlabel='')    
    plt.show()

def _plot_violin_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.violinplot(x=cat, y=quant, data=train, split=True, 
                           ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()
    
def _plot_swarm_grid_with_color(train, target, cat_vars, quant_vars):
    cols = len(cat_vars)
    for quant in quant_vars:
        _, ax = plt.subplots(nrows=1, ncols=cols, figsize=(16, 4), sharey=True)
        for i, cat in enumerate(cat_vars):
            sns.swarmplot(x=cat, y=quant, data=train, ax=ax[i], hue=target, palette="Set2")
            ax[i].set_xlabel('')
            ax[i].set_ylabel(quant)
            ax[i].set_title(cat)
        plt.show()

def idf(word, document_series):
    n_occurences = sum([1 for doc in document_series if word in doc])
    
    return np.log(len(document_series) / n_occurences)

def generate_tf_idf_tfidf_dataframe(word_list, document_series):
    word_freq_df = (pd.DataFrame({'raw_count': word_list.value_counts()})\
                    .assign(frequency=lambda df: df.raw_count / df.raw_count.sum())\
                    .assign(augmented_frequency=lambda df: df.frequency / df.frequency.max()))
    
    word_freq_df = word_freq_df.reset_index()
    word_freq_df = word_freq_df.rename(columns={'index' : 'word'})

    word_freq_df['idf'] = word_freq_df.word.apply(idf, document_series=document_series)
    word_freq_df['tf_idf'] = word_freq_df.frequency * word_freq_df.idf
    
    return word_freq_df