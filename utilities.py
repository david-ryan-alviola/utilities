import pandas as pd
import os

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
    print(f"corr:  {correlation}, p:  {p_value}, a:  {alternative_hypothesis}")
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