import pandas as pd
import sys
from keras import regularizers
from preprocessing import *
from PaddingModel import PaddingModel
from OutputInputModel import OutputInputModel
from keras.models import load_model
import cProfile


def get_config(attr, output):
    return {
    "batch_size": 35,
    "attr": attr,
    "target_variable": output,
    "time_steps": 36, #use 6 last hours
    "forecast_steps": 36, # to predict 6 next hour
    "num_epochs": 1,
    "skip_steps": 6,
    "hidden_size": 500,
    "load_file": None,
    "regularizer": regularizers.l2(0.01)
    }

def main():
    if len(sys.argv) == 1:
        print("usage : %s pathtodataset" % (sys.argv[0]))
        return 1
    df = read_file(sys.argv[1])

    #df = arrange_data(df)
    #output_list = list(df.drop(["Time", "month", "hour", "day"], 1))
    output_list = list(df.drop(["Time"], 1))
    df, copied_attr = copy_target(df, output_list)
    pd.set_option('display.max_columns', None)
    copied_attr.append("Time")
    attr_list = list(df.drop(copied_attr, 1))
    df_mod = clean_data(df, attr_list)
    copied_attr.remove("Time")

    #print(df_mod.min())
    #print(df_mod.max())

    training_set, validation_set, test_set = separate_set_seq(df_mod, 80, 19, 1)

    config = get_config(attr_list, copied_attr)

    if config["load_file"] is None:
        model = OutputInputModel(config)
        model.train(training_set, validation_set, plotLoss=False)
        #apr = cProfile.Profile()
        #pr.enable()
        model.output_as_input_testing(validation_set)
        #pr.disable()
        # after your program ends
        #pr.print_stats(sort="line")
        exit(0)

        model.save("/cluster/home/arc/bjl31/full_model.h5")
    else:
        raise Exception("Not implemented")
        model = load_model(config["load_file"])


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