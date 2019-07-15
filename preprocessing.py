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

def min_max(df):
    df_min = (df - df.min()) / (df.max() - df.min())
    return df_min

def separate_set_seq(df, train_fraction=70, valid_fraction=10, test_fraction=20):
    """Separate set without sampling. fraction params must sum up to 100"""
    len_ = len(df)
    idx_train = len_ * train_fraction // 100
    idx_valid = idx_train + len_ * valid_fraction // 100

    return df[:idx_train].reset_index(), df[idx_train:idx_valid].reset_index(), df[idx_valid:].reset_index()

def clean_data(df, columns):
    pd.set_option('display.width', 2000)

    df_cp = df[[x for x in list(df) if x not in columns]]

    df_mod = df.drop(list(df_cp), 1).apply(pd.to_numeric, 1, errors="coerce")
    df_mod = normalize(df_mod)
    #df_mod = min_max(df_mod)
    df_mod = df_mod.join(df_cp)
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
