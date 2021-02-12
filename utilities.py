def evaluate_hypothesis_ttest(null_hypothesis, alternative_hypothesis, p_value, t_value, alpha = .05, tails = "two"):
    """
    Utility function to evaluate T-test hypothesis

    Evaluates whether or not the null hypothesis can be rejected after performing T-test analysis.

    Parameters
    ----------
    null_hypothesis : str
        The null hypothesis being tested
    alternative_hypothesis : str
        The alternative hypothesis being tested
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
    
    Returns
    -------
    None
        Prints the result of the evaluation
    """
    def fail_to_reject_null_hypothesis():
        print(f"We fail to reject the null hypothesis:  {null_hypothesis}")

    def reject_null_hypothesis():
        print("We reject the null hypothesis.")
        print(f"We move forward with the alternative hypothesis:  {alternative_hypothesis}")

    print(f"t:  {t_value}, p:  {p_value}, a:  {alpha}")
    print()

    if tails == "two":

        if p_value < alpha:
            reject_null_hypothesis()
        else:
            fail_to_reject_null_hypothesis()
    else:

        if (p_value / 2) < alpha:

            if (tails == "greater"):
                if (t_value > 0):
                    reject_null_hypothesis()
            elif (tails == "less"):
                if (t_value < 0):
                    reject_null_hypothesis()
            else:
                raise ValueError("tails parameter only accepts:  'two', 'greater', 'less'")
        else:
            fail_to_reject_null_hypothesis()

def generate_db_url(user, password, host, db_name):
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
    
    Returns
    -------
    str
        URL in this format:  mysql+pymysql://user:password@host/db_name
    """
    return f"mysql+pymysql://{user}:{password}@{host}/{db_name}"