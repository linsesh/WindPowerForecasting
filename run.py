import pandas as pd
import sys
from keras import regularizers
from preprocessing import *
from PaddingModel import PaddingModel
from OutputInputModel import OutputInputModel
from keras.models import load_model
import cProfile
from time import gmtime, strftime

#to put in a json file, more pratical
def get_config(attr, output):
    return {
    "batch_size": 35,
    "attr": attr,
    "target_variable": output,
    "time_steps": 36, #use 12 last hours
    "forecast_steps": 72, # to predict 6 next hour
    "num_epochs": 15,
    "skip_steps": 6,
    "hidden_size": 500,
    "load_file": None,
    "regularizer": regularizers.l2(0.01) #for now
    }

def main():
    if len(sys.argv) == 1:
        print("usage : %s pathtodataset" % (sys.argv[0]))
        return 1

    timetoday = (strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    print("run of %s" % (timetoday))

    df = read_file(sys.argv[1])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

    #df = collapse_data(df, window=3)
    df = arrange_data(df)
    #output_list = ["Wind average [m/s]"]
    output_list = list(df.drop(["Time", "month", "hour", "day"], 1))
    #output_list = list(df.drop(["Time"], 1))

    df, copied_attr = copy_target(df, output_list)

    copied_attr.append("Time")
    attr_list = list(df.drop(copied_attr, 1))

    training_set, validation_set, test_set = separate_set_seq(df, 90, 9, 1)

    training_set = clean_data(training_set, attr_list)
    validation_set = clean_data(validation_set, attr_list)

    copied_attr.remove("Time")
    training_set.reset_index(drop=True, inplace=True)
    validation_set.reset_index(drop=True, inplace=True)

    config = get_config(attr_list, copied_attr)

    if config["load_file"] is None:
        model = PaddingModel(config)
        model.train(training_set, validation_set, plotLoss=True)
        #apr = cProfile.Profile()
        #pr.enable()
        model.test_variables_mutivariate_model(validation_set, ["target_Wind average [m/s]"], "/cluster/home/arc/bjl31/propre/predicted-truth-%s" % (timetoday))
        model.save("/cluster/home/arc/bjl31/propre/model_of-%s.h5" % timetoday)
        #model.output_as_input_testing(validation_set)
        #pr.disable()
        # after your program ends
        #pr.print_stats(sort="line")
        exit(0)

    else:
        model = load_model(config["load_file"])
        model = PaddingModel(config, model)
        model.test_one_variable_mutivariate_model(validation_set)


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
