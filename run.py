import pandas as pd
import sys
from preprocessing import *
from PaddingModel import PaddingModel
from OutputInputModel import OutputInputModel
from PersistenceModel import PersistenceModel
from keras.models import load_model
import cProfile
from time import gmtime, strftime
import numpy as np
from arima import fit_arima
from config import get_config

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
    #df = arrange_data(df)
    #output_list = ["Wind average [m/s]"]
    #output_list = list(df.drop(["Time", "month", "hour", "day"], 1))
    output_list = list(df.drop(["Time"], 1))

    df, copied_attr = copy_target(df, output_list)

    copied_attr.append("Time")
    attr_list = list(df.drop(copied_attr, 1))

    training_set = df

    training_set = clean_data(training_set, attr_list)

    copied_attr.remove("Time")
    training_set.reset_index(drop=True, inplace=True)

    config = get_config(attr_list, copied_attr)

    results = []
    tested_variables = ["target_Wind average [m/s]", "target_Power average [kW]"]
    if config["load_file"] is None:
        for run in range(5):
            idx = len(training_set) * run // 5
            limit = len(training_set) * (run + 1) // 5
            validation_set = training_set.loc[np.arange(idx, limit).tolist()]
            indices = []
            indices.extend(np.arange(limit, len(training_set)).tolist())
            indices.extend(np.arange(idx))
            train_set = training_set.loc[indices]
            train_set.reset_index(drop=True, inplace=True)
            print(len(train_set))
            validation_set.reset_index(drop=True, inplace=True)
            model = OutputInputModel(config)
            model.train(train_set, validation_set, plotLoss=True)
            #apr = cProfile.Profile()
            #pr.enable()
            res = model.output_as_input_testing(validation_set, tested_variables, forecast_steps=72) #"/cluster/home/arc/bjl31/propre/predicted-truth-%s" % (timetoday)
            results.append(res)
        for i in range(len(tested_variables)):
            var_res = [x[i] for x in results]
            mse = [x[0] for x in var_res]
            mae = [x[1] for x in var_res]
            print(mae)
            print(mse)
            print(sum(mae) / float(len(mae)))
            print(sum(mse) / float(len(mse)))
        #model.save("/cluster/home/arc/bjl31/propre/model_of-%s.h5" % timetoday)
        #model.output_as_input_testing(validation_set)
        #pr.disable()
        # after your program ends
        #pr.print_stats(sort="line")

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
