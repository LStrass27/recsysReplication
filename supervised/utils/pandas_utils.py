def group_categoricals_tail(data, categoricals, percentile=0.01, grouped=True):
    """Takes a pandas DataFrame and groups the categorical columns modes which, 
    grouped, account for less than percentile overall.
    !!!WARNING!!!
    It overrides the DataFrame object!

    Args:
        - data (pandas DataFrame): pandas DataFrame to modify
        - categoricals (list): list of the columns name of categorical features
        - percentile (float, default = 0.99): number between 0 and 1
    """
    #print(grouped)

    import numpy as np
    for i in range(len(categoricals)):
        col = categoricals[i]
        if grouped:
            '''
            cum_sum = data[col].value_counts().cumsum().reset_index().rename(columns={"index":col, col:"cumsum"})
            print(cum_sum.columns)
            cum_sum["cumsum"] = cum_sum["cumsum"] / data.shape[0]
            print(cum_sum)
            others = cum_sum.loc[cum_sum["cumsum"] >= 1 - percentile, col].tolist()
            '''
            value_counts = data[col].value_counts().cumsum()
            cum_sum = value_counts.reset_index()
            cum_sum.columns = [col, "cumsum"]
            #print(cum_sum.columns)
            cum_sum["cumsum"] = cum_sum["cumsum"] / data.shape[0]
            others = cum_sum.loc[cum_sum["cumsum"] >= 1 - percentile, col].tolist()

        else:
            count = data[col].value_counts().cumsum().reset_index().rename(columns={"index":col, col:"count"})
            count["count"] = count["count"] / data.shape[0]
            others = cum_sum.loc[cum_sum["cumsum"] < percentile, col].tolist()
        name = "_".join(["others", col])
        data[col] = np.where(data[col].isin(others), name, data[col])