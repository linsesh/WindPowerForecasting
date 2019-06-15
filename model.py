import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam



class PandasBatchGenerator(object):

    def __init__(self, data, num_steps, attr_column, target_column, batch_size, skip_step):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.attr_col = attr_column
        self.target_col = target_column

        self.current_idx = 0
        self.skip_step = skip_step
        self.idx_errors = []


    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, len(self.attr_col)))
        y = np.zeros((self.batch_size, self.num_steps, 1))
        while True:
            i = 0
            while i < self.batch_size:
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0

                try:
                    x[i, :, :] = self.data.loc[self.current_idx:self.current_idx + self.num_steps - 1, self.attr_col].values
                    y[i, :, :] = self.data.loc[self.current_idx + 1:self.current_idx + self.num_steps, self.target_col].values
                except Exception as e:
                    print(e)
                    self.idx_errors.append(self.current_idx)
                    i = i - 1

                self.current_idx += self.skip_step
                i = i + 1
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

    df = df[0:50000]
    df_mod = df.drop("Time", 1).apply(pd.to_numeric, 1, errors="coerce")
    df_mod = normalize(df_mod)
    df_mod = df_mod.assign(Time=df["Time"])
    df_mod.dropna(inplace=True)

    batch_size = 100
    attr = list(df_mod.drop("Time", 1))
    time_steps = 6
    generator = PandasBatchGenerator(df_mod, time_steps, attr, ["Power average [kW]"], batch_size, time_steps)

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_steps, len(attr))))
    model.add(Dense(1))
    model.add(Activation("tanh"))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    model.fit_generator(generator.generate(), len(df_mod)//batch_size, 1)
    print("nb inputs skipped = %d" % (len(generator.idx_errors)))
    #model.evaluate_generator(generator, )

if __name__ == "__main__":
    main()

#df = df[0:10000]
#df = df.sample(n=10000)
#df["Wind average [m/s]"].plot(kind="hist")
#plt.show()
