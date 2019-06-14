import sys
import pandas as pd
import matplotlib.pyplot as plt

def normalize(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    print(df_norm.mean())
    print(df_norm.std())
    df_std = (df - df.mean()) / df.std()
    print(df_std.mean())
    print(df_std.std())
    return df_std

def main():
    if len(sys.argv) == 1:
        print("usage : %s pathtodataset" % (sys.argv[0]))
        return 1
    try:
        df = pd.read_excel(sys.argv[1])
    except Exception as e:
        print("Could not open %s" % (sys.argv[1]))
        print(e)
        return 1

    #df = df.sample(n=10000)
    #df["Wind average [m/s]"].plot(kind="hist")
    #plt.show()
    df_mod = df.drop("Time", 1).apply(pd.to_numeric, 1, errors="coerce")
    df_mod.dropna(inplace=True)
    df_mod = normalize(df_mod)
    df_mod = df_mod.assign(Time=df["Time"])

if __name__ == "__main__":
    main()