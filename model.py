import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.utils import plot_model


class PandasBatchGenerator(object):

    def __init__(self, data, num_steps, attr_column, target_column, batch_size, skip_step, returnSequence=False):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.attr_col = attr_column
        self.target_col = target_column

        self.current_idx = 0
        self.skip_step = skip_step
        self.returnSeq = returnSequence
        self.idx_errors = []


    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps, len(self.attr_col)))
        if not self.returnSeq:
            y = np.zeros((self.batch_size, 1))
        else:
            raise Exception("not implemented")
            y = np.zeros((self.batch_size, self.num_steps, 1))
        while True:
            i = 0
            while i < self.batch_size:
                if self.current_idx + self.num_steps >= len(self.data):
                    self.current_idx = 0

                try:
                    x[i, :, :] = self.data.loc[self.current_idx:self.current_idx + self.num_steps - 1, self.attr_col].values
                    if not self.returnSeq:
                        y[i, 0] = self.data.loc[self.current_idx + self.num_steps, self.target_col].values
                    else:
                        y[i, :, :] = self.data.loc[self.current_idx + 1:self.current_idx + self.num_steps, self.target_col].values
                except Exception as e:
                    print(e)
                    self.idx_errors.append(self.current_idx)
                    i = i - 1

                self.current_idx += self.skip_step
                i = i + 1
            yield x, y

def separate_set_seq(df, train_fraction=70, valid_fraction=10, test_fraction=20):
    """Separate set without sampling. fraction params must sum up to 100"""
    len_ = len(df)
    idx_train = len_ * train_fraction // 100
    idx_valid = idx_train + len_ * valid_fraction // 100

    return df[:idx_train].reset_index(), df[idx_train:idx_valid].reset_index(), df[idx_valid:].reset_index()

def get_config(df):
    return {
    "batch_size": 50,
    "attr": list(df.drop("Time", 1)),
    "time_steps": 12,
    "num_epochs": 30,
    "skip_steps": 1,
    "hidden_size": 500
    }


def normalize(df):
    df_norm = (df - df.mean()) / (df.max() - df.min())
    #print(df_norm.mean())
    #print(df_norm.std())
    df_std = (df - df.mean()) / df.std()
    #print(df_std.mean())
    #print(df_std.std())
    return df_std

#def test_model(model, test_set):

def get_trained_model(training_set, validation_set, config, loadFile=None, plotLoss=False):
    if loadFile:
        return load_model(loadFile)

    training_generator = PandasBatchGenerator(training_set, config["time_steps"], config["attr"],
                                              ["Power_average"], config["batch_size"], config["skip_steps"])
    validation_generator = PandasBatchGenerator(validation_set, config["time_steps"], config["attr"],
                                                ["Power_average"], config["batch_size"], config["skip_steps"])

    model = Sequential()
    model.add(LSTM(config["hidden_size"], input_shape=(config["time_steps"], len(config["attr"])), unroll=True, return_sequences=True))
    model.add(LSTM(config["hidden_size"], unroll=True))
    model.add(Dense(units=100))
    model.add(Activation("tanh"))
    #model.add(LSTM(500, stateful=True, batch_input_shape=(config["batch_size"], config["time_steps"], len(config["attr"]))))
    #model.add(LSTM(500, stateful=True))
    model.add(Dense(1))
#    model.add(Activation("tanh"))

    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    history = model.fit_generator(training_generator.generate(), len(training_set) // (config["batch_size"]*(config["time_steps"] + config["skip_steps"])),
                                  config["num_epochs"], validation_data=validation_generator.generate(),
                                  validation_steps=len(validation_set) // (config["batch_size"]*(config["time_steps"] + config["skip_steps"])), shuffle=False)
    print("nb inputs skipped = %d" % (len(training_generator.idx_errors)))
    if plotLoss:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("/cluster/home/arc/bjl31/loss.png")

    return model

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

    df_mod = df.drop("Time", 1).apply(pd.to_numeric, 1, errors="coerce")
    df_mod = normalize(df_mod.drop("Power average [kW]", 1))
    df_mod = df_mod.assign(Time=df["Time"], Power_average=df["Power average [kW]"])
    print(df_mod)
    df_mod.dropna(inplace=True)
    print(df_mod)
    exit(0)

    training_set, validation_set, test_set = separate_set_seq(df_mod)

    config = get_config(df_mod)

    model = get_trained_model(training_set, validation_set, config, plotLoss=True)

    #test_model(model, test_set)

    test_generator = PandasBatchGenerator(test_set, config["time_steps"], config["attr"],
                                          ["Power_average"], 1, config["skip_steps"])
    test_generator_bis = PandasBatchGenerator(test_set, config["time_steps"], config["attr"],
                                          ["Power_average"], 1, config["skip_steps"])

    ev = model.evaluate_generator(test_generator_bis.generate(), len(test_set) // (config["batch_size"]*config["skip_steps"]))
    print("error on test set: %d" % ev)

    for i in range(1):
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        print(test_set[test_generator.current_idx:test_generator.current_idx + test_generator.num_steps + 1])
        inp, out = next(test_generator.generate())
        y = model.predict(inp, verbose=1)
        print(y)
        print(out)
    model.save("/cluster/home/arc/bjl31/full_model.h5")

if __name__ == "__main__":
    main()

#df = df[0:10000]
#df = df.sample(n=10000)
#df["Wind average [m/s]"].plot(kind="hist")
#plt.show()
