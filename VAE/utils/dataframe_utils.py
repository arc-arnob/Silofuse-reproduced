import pandas as pd

def split_columns(df: pd.DataFrame, save_path=None, diabetes_data=False):
    """
    Split the input DataFrame into two parts:
    - all columns except the last one
    - the last column

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame, pd.Series: A DataFrame with all columns except the last one, and the last column as a Series.
    """
    # All columns except the last one

    all_except_last = df.iloc[:, :-1]

    # The last column
    if diabetes_data:
        last_column = df.iloc[:, -1].map({"b'tested_positive'": 1, "b'tested_negative'": 0})
    else:    
        last_column = df.iloc[:, -1]
    all_except_last.to_csv(save_path, index=False)
    return all_except_last, last_column