def _fail_to_reject_null_hypothesis(null_hypothesis):
    return f"We fail to reject the null hypothesis:  {null_hypothesis}"

def _reject_null_hypothesis(alternative_hypothesis):
    return f"We reject the null hypothesis. We move forward with the alternative hypothesis:  {alternative_hypothesis}"

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

    result = {'t' :  t_value, 'p' :  p_value, 'a' :  alpha}

    print("------------------------------------------")
    print(f"t:  {t_value}, p:  {p_value}, a:  {alpha}")
    print()

    if tails == "two":

        if p_value < alpha:
            result['message'] = _reject_null_hypothesis(alternative_hypothesis)
            result['reject_null'] = True

        else:
            result['message'] = _fail_to_reject_null_hypothesis(null_hypothesis)
            result['reject_null'] = False

    else:

        if (p_value / 2) < alpha:

            if (tails == "greater"):
                if (t_value > 0):
                    result['message'] = _reject_null_hypothesis(alternative_hypothesis)
                    result['reject_null'] = True

                else:
                    result['message'] = _fail_to_reject_null_hypothesis(null_hypothesis)
                    result['reject_null'] = False

            elif (tails == "less"):
                if (t_value < 0):
                    result['message'] = _reject_null_hypothesis(alternative_hypothesis)
                    result['reject_null'] = True
                
                else:
                    result['message'] = _fail_to_reject_null_hypothesis(null_hypothesis)
                    result['reject_null'] = False

            else:
                raise ValueError("tails parameter only accepts:  'two', 'greater', 'less'")

        else:
            result['message'] = _fail_to_reject_null_hypothesis(null_hypothesis)
            result['reject_null'] = False

    print(result['message'])
    print("------------------------------------------")

    return result

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

    print("------------------------------------------")
    print(f"corr:  {correlation}, p:  {p_value}, a:  {alpha}")
    print()

    result = {'corr' :  correlation, 'p' :  p_value, 'a' :  alpha}

    if correlation > 1 or correlation < -1:
        raise ValueError("The correlation must be between -1 and 1")
    else:
        if p_value < alpha:
            result['reject_null'] = True
            result['message'] = _reject_null_hypothesis(alternative_hypothesis)
        else:
            result['reject_null'] = False
            result['message'] = _fail_to_reject_null_hypothesis(null_hypothesis)

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