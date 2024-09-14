import os
import sys
import pandas as pd

def check_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_df = missing_values[missing_values > 0].reset_index()
    missing_values_df.columns = ['column_name', 'missing_count']
    return missing_values_df

def missing_percentage(df):
    missing_percentatges = df.isnull().mean() * 100 
    column_with_missing_values = missing_percentatges[missing_percentatges > 30].index.tolist()
    return column_with_missing_values

def get_numeric_columns(df):
    """
    Get a list of column names with numeric data types from a DataFrame.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - numeric_columns: List of column names with numeric data types
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numeric_columns
    
def check_numeric_anomalies(df, column, lower_bound=None, upper_bound=None):
    """
    Check for numeric anomalies in a specific column of a DataFrame and return a summary.
    
    Parameters:
    - df: Pandas DataFrame
    - column: The specific column to check
    - lower_bound: Lower bound for numeric anomalies (optional)
    - upper_bound: Upper bound for numeric anomalies (optional)
    
    Returns:
    - str or DataFrame: Success message or summary of anomalies
    """
    if df[column].dtype not in ['int64', 'float64']:
        return f"Error: Column {column} is not numeric."
    
    if lower_bound is not None and upper_bound is not None:
        anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    else:
        anomalies = df[~df[column].apply(lambda x: isinstance(x, (int, float)))]
    
    if anomalies.empty:
        return "Success: No anomalies detected."
    else:
        anomalies_summary = pd.DataFrame({
            'Column Name': [column],
            'Number of Anomalies': [len(anomalies)]
        })
        return anomalies_summary