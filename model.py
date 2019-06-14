import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PandasBatchGenerator(object):

    def __init__(self, data, num_steps, attr_column, target_column, batch_size, skip_step):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.attr_col = attr_column
        self.target_col = target_column

        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, len(self.attr_col)))
        y = np.zeros((self.batch_size, self.num_steps))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0

                try:
                    x[i, :, :] = self.data.loc[self.current_idx:self.current_idx + self.num_steps - 1, self.attr_col].values
                    y[i, :] = self.data.loc[self.current_idx + 1:self.current_idx + self.num_steps, self.target_col].values.reshape(self.num_steps)
                except Exception as e:
                    print(e)
                    print(self.data[self.current_idx:])

                self.current_idx += self.skip_step
            yield x, y

def normalize(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    #print(df_norm.mean())
    #print(df_norm.std())
    df_std = (df - df.mean()) / df.std()
    #print(df_std.mean())
    #print(df_std.std())
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

    df = df[0:10000]
    df_mod = df.drop("Time", 1).apply(pd.to_numeric, 1, errors="coerce")
    df_mod.dropna(inplace=True)
    df_mod = normalize(df_mod)
    df_mod = df_mod.assign(Time=df["Time"])

    generator = PandasBatchGenerator(df_mod, 6, list(df_mod.drop("Time", 1)), ["Power average [kW]"], 50, 6)

if __name__ == "__main__":
    main()

#df = df[0:10000]
#df = df.sample(n=10000)
#df["Wind average [m/s]"].plot(kind="hist")
#plt.show()
