import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Lambda, LSTM
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.utils import plot_model
from batch_generator import PandasBatchGenerator
from preprocessing import *

def get_config(df):
    return {
    "batch_size": 50,
    "attr": list(df.drop(["Time", "Power_average_output"], 1)),
    "time_steps": 18, #use 3 last hours
    "forecast_steps": 12, # to predict 2 next hour
    "num_epochs": 20,
    "skip_steps": 6,
    "hidden_size": 500,
    "load_file": None
    }

#def test_model(model, test_set):

def plot_infos(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    try:
        plt.savefig("/cluster/home/arc/bjl31/loss.png")
    except Exception as e:
        print(e)

def get_trained_model(training_set, validation_set, config, plotLoss=False):

    training_generator =  PandasBatchGenerator(training_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                              ["Power_average_output"], config["batch_size"], config["skip_steps"])
    validation_generator = PandasBatchGenerator(validation_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                                ["Power_average_output"], config["batch_size"], config["skip_steps"])

    #inp, out =  next(training_generator.generate())

    model = Sequential()
    model.add(LSTM(config["hidden_size"], input_shape=(config["time_steps"] + config["forecast_steps"], len(config["attr"])), unroll=True, return_sequences=True))
    model.add(LSTM(config["hidden_size"], unroll=True, return_sequences=True))
    model.add(Lambda(lambda x: x[:, -config["forecast_steps"]:, :]))
    model.add(Dense(units=100))
    model.add(Activation("tanh"))
    model.add(Dense(1))

    #print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_absolute_error"])
    print(model.summary())
    history = model.fit_generator(training_generator.generate(), len(training_set) // ((config["batch_size"]*config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]),
                                  config["num_epochs"], validation_data=validation_generator.generate(),
                                  validation_steps=len(validation_set) // ((config["batch_size"]*config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]), shuffle=False,
 )
    print("nb inputs skipped = %d" % (len(training_generator.idx_errors)))

    if plotLoss:
        plot_infos(history)

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

    df_mod = arrange_data(df)
    df_mod = clean_data(df_mod)
    training_set, validation_set, test_set = separate_set_seq(df_mod)

    config = get_config(df_mod)

    if config["load_file"] is None:
        model = get_trained_model(training_set, validation_set, config, plotLoss=True)
    else:
        model = load_model(config["load_file"])


    test_generator = PandasBatchGenerator(test_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                          ["Power_average_output"], 1, config["skip_steps"])
    test_generator_bis = PandasBatchGenerator(test_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                          ["Power_average_output"], 1, config["skip_steps"])

    ev = model.evaluate_generator(test_generator_bis.generate(),
                                  len(test_set) // ((config["batch_size"]*config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]),
                                  verbose=1)
    print("error on test set: [%s]" % ', '.join(map(str, ev)))

    print("example of prediction :")
    for i in range(200):
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        #print(test_set[test_generator.current_idx:test_generator.current_idx + test_generator.num_steps + test_generator.num_padding])
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
    #df["Power average [kW]"].plot(kind="hist")
    #df_mod["Power_average"].plot(kind="hist")
    ##df.boxplot(column=["Wind average [m/s]"])
    #plt.show()
    #exit(0)