import pandas as pd
import os

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