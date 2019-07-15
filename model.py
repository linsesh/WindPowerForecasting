import pandas as pd
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Lambda, LSTM
from keras import regularizers
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.utils import plot_model
from preprocessing import *
from testing import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def get_config(attr, output):
    return {
    "batch_size": 35,
    "attr": attr,
    "target_variable": output,
    "time_steps": 36, #use 6 last hours
    "forecast_steps": 36, # to predict 6 next hour
    "num_epochs": 30,
    "skip_steps": 6,
    "hidden_size": 500,
    "load_file": None,
    "regularizer": regularizers.l2(0.01)
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
                                              config["target_variable"], config["batch_size"], config["skip_steps"])
    validation_generator = PandasBatchGenerator(validation_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                                config["target_variable"], config["batch_size"], config["skip_steps"])

    #inp, out =  next(training_generator.generate())
    input_shape = (config["time_steps"] + config["forecast_steps"], len(config["attr"]))
    model = Sequential()
    #model.add(Dense(100, kernel_regularizer=config["regularizer"], input_shape=input_shape))
    #model.add(Activation("tanh"))
    model.add(LSTM(config["hidden_size"], unroll=True, return_sequences=True, kernel_regularizer=config["regularizer"], input_shape=input_shape))
    model.add(LSTM(config["hidden_size"], unroll=True, return_sequences=True, kernel_regularizer=config["regularizer"]))
    model.add(Lambda(lambda x: x[:, -config["forecast_steps"]:, :]))
    model.add(Dense(len(config["target_variable"]), kernel_regularizer=config["regularizer"]))

    #print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_absolute_error"])
    print(model.summary())
    history = model.fit_generator(training_generator.generate(), len(training_set) // ((config["batch_size"]*config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]),
                                  config["num_epochs"], validation_data=validation_generator.generate(),
                                  validation_steps=len(validation_set) // ((config["batch_size"]*config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]), shuffle="batch",
 )
    print("nb inputs skipped = %d" % (len(training_generator.idx_errors)))

    if plotLoss:
        plot_infos(history)

    return model


def main():
    if len(sys.argv) == 1:
        print("usage : %s pathtodataset" % (sys.argv[0]))
        return 1
    df = read_file(sys.argv[1])

    df = arrange_data(df)
    output_list = list(df.drop(["Time", "month", "hour", "day"], 1))
    df, copied_attr = copy_target(df, output_list)
    pd.set_option('display.max_columns', None)
    copied_attr.append("Time")
    attr_list = list(df.drop(copied_attr, 1))
    df_mod = clean_data(df, attr_list)
    copied_attr.remove("Time")
    print(df_mod.min())
    print(df_mod.max())

    training_set, validation_set, test_set = separate_set_seq(df_mod, 80, 19, 1)

    config = get_config(attr_list, copied_attr)

    if config["load_file"] is None:
        model = get_trained_model(training_set, validation_set, config, plotLoss=True)
    else:
        model = load_model(config["load_file"])

    predictions = []
    observations = []
    for n in range((len(validation_set) - 36) // 6):
        x = np.full((1, config["forecast_steps"] + config["time_steps"], len(config["attr"])), 0.0)
        x[0,:config["time_steps"],:] = validation_set.loc[n * 6:n * 6 + 35, attr_list]
        output = model.predict(x)
        for t in range(36):
            yhat = output[0][t][config["target_variable"].index("target_Power average [kW]")]
            predictions.append(yhat)
            obs = validation_set.loc[n * 6 + t, "target_Power average [kW]"]
            observations.append(obs)
            #print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(observations, predictions)
    print('Test MSE: %.3f' % error)
    mae = mean_absolute_error(observations, predictions)
    print('Test MAE: %.3f' % mae)

    exit(0)

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