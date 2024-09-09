def split_columns(df: pd.DataFrame):
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
    last_column = df.iloc[:, -1]
    all_except_last.to_csv("../data/interim/bank_no_label.csv", index=False)
    return all_except_last, last_column