def feature_selection(df):
    """
    Perform feature selection on the dataset.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Dataset with selected features.
    """

    # Example: Select specific columns (features) for analysis
    selected_features = ['PU', 'PEU', 'FSC', 'SP', 'TP', 'IB']  # Replace with actual feature names
    df_selected = df['AUB']

    return df_selected