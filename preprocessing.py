import pandas as pd
from datetime import datetime

def read_file(name):
    try:
        df = pd.read_excel(name)
    except Exception as e:
        print("Could not open %s" % name)
        print(e)
        raise Exception("Failure, closing program")
    return df

def normalize(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    #print(df_norm.mean())
    #print(df_norm.std())
    df_std = (df - df.mean()) / df.std()
    #print(df_std.mean())
    #print(df_std.std())
    return df_std

#a class wrapping or inheriting dataframe would have been better software engineering
#pandas df cannot be hashed, so it cant be used as key in a dictionary
def normalize_from_df(data, df):
    ret = []
    if normalize_from_df.dataframes is not None:
        _cache = normalize_from_df.dataframes
        ret = [(data[i] - _cache[i][0]) / _cache[i][1] for i in range(len(data))]
    else:
        for i in range(len(data)):
            ret.append((data[i] - df.iloc[:, i].mean()) / df.iloc[:, i].std())
        normalize_from_df.dataframes = [(df[column].mean(), df[column].std()) for column in df]
    return ret
normalize_from_df.dataframes = None

def min_max(df):
    df_min = (df - df.min()) / (df.max() - df.min())
    return df_min

def separate_set_seq(df, train_fraction=70, valid_fraction=10, test_fraction=20):
    """Separate set without sampling. fraction params must sum up to 100"""
    len_ = len(df)
    idx_train = len_ * train_fraction // 100
    idx_valid = idx_train + len_ * valid_fraction // 100

    return df[:idx_train].reset_index(drop=True), df[idx_train:idx_valid].reset_index(drop=True), df[idx_valid:].reset_index(drop=True)

def collapse_data(df, window):
    """averages window size data into one row"""
    index = 0
    new_df = pd.DataFrame()
    cols = [x for x in list(df.columns) if x not in ["Nacelle pos. [°] (Wind direction)", "Time"]]
    while index + window < len(df):
        collapsed_row = df.loc[index:index + window - 1, cols].mean()
        wdirs = df.loc[index:index + window - 1, "Nacelle pos. [°] (Wind direction)"].to_numpy()
        sector4 = True if next((elem for elem in wdirs if elem > 270), None) is not None else False
        sum = 0
        for i in range(window):
            sum += wdirs[i]
            if sector4 and wdirs[i] < 180:
                sum += 360
        sum /= window
        if sum > 360:
            sum -= 360
        collapsed_row.loc["Nacelle pos. [°] (Wind direction)"] = sum
        collapsed_row.loc["Time"] = df.loc[index + window // 2, "Time"]
        new_df = new_df.append(collapsed_row, ignore_index=True)
        index += window
    new_df = new_df[df.columns.to_list()] #reorder indexes of columns
    print(df.loc[:, "Nacelle pos. [°] (Wind direction)"].mean())
    print(new_df.loc[:, "Nacelle pos. [°] (Wind direction)"].mean())
    print(df)
    print(new_df)
    return new_df

def clean_data(df, columns):

    df_cp = df[[x for x in list(df) if x not in columns]]

    df_mod = df.drop(list(df_cp), 1).apply(pd.to_numeric, 1, errors="coerce")
    df_mod = normalize(df_mod)
    #df_mod = min_max(df_mod)
    df_mod = df_mod.join(df_cp)
    print(df_mod.isna().sum())
    df_mod.dropna(inplace=True)
    return df_mod

def copy_target(df, targets):
    names = {}
    for t in targets:
        name = "target_" + t
        names[name] = df[t]
    df = df.assign(**names)
    return df, [*names]

def arrange_data(df):
    df["month"] = df.apply(lambda row: datetime.strptime(row.Time, '%m/%d/%Y  %I:%M %p').month, axis=1)
    df["hour"] = df.apply(lambda row: datetime.strptime(row.Time, '%m/%d/%Y  %I:%M %p').hour, axis=1)
    df["day"] = df.apply(lambda row: datetime.strptime(row.Time, '%m/%d/%Y  %I:%M %p').day, axis=1)
    return df
