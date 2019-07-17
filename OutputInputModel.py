from base_model import *
from preprocessing import normalize_from_df
import cProfile

class OutputInputModel(base_model):

    def __init__(self, config):
        self.config = config
        input_shape = (config["time_steps"], len(config["attr"]))

        self.model = Sequential()
        # model.add(Dense(100, kernel_regularizer=config["regularizer"], input_shape=input_shape))
        # model.add(Activation("tanh"))
        self.model.add(
            LSTM(config["hidden_size"], unroll=True, return_sequences=True, kernel_regularizer=config["regularizer"],
                 input_shape=input_shape))
        self.model.add(
            LSTM(config["hidden_size"], unroll=True, return_sequences=False, kernel_regularizer=config["regularizer"]))
        self.model.add(Dense(len(config["target_variable"]), kernel_regularizer=config["regularizer"]))

        # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_absolute_error"])
        print(self.model.summary())



    def train(self, training_set, validation_set, plotLoss=True):
        model = self.model
        config = self.config

        training_generator = PandasBatchGenerator(training_set, config["time_steps"], None, config["attr"],
                                                  config["target_variable"], config["batch_size"], config["skip_steps"])
        validation_generator = PandasBatchGenerator(validation_set, config["time_steps"], None, config["attr"],
                                                    config["target_variable"], config["batch_size"], config["skip_steps"])        # inp, out =  next(training_generator.generate())

        history = model.fit_generator(training_generator.generate(), len(training_set) // (
                    (config["batch_size"] * config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]),
                                      config["num_epochs"], validation_data=validation_generator.generate(),
                                      validation_steps=len(validation_set) // (
                                                  (config["batch_size"] * config["skip_steps"]) + config["time_steps"] +
                                                  config["forecast_steps"]), shuffle=False,
                                      )
        print("nb inputs skipped = %d" % (len(training_generator.idx_errors)))

        if plotLoss:
            self.plot_infos(history)

    def output_as_input_testing(self, validation_set):
        model = self.model
        config = self.config

        predictions = []
        var_prediction = []
        truth_values = []
        variables = validation_set[config["target_variable"]]
        skip = config["skip_steps"]
        for n in range((len(validation_set) - (config["forecast_steps"] + config["time_steps"])) // skip):
            x = np.zeros((1, config["time_steps"], len(config["attr"])))
            for t in range(config["forecast_steps"]):
                x[0, :config["time_steps"] - t, :] = validation_set.loc[n * skip + t:n * skip + config["time_steps"] - 1,
                                                     config["attr"]]
                if t > 0:
                    #print(predictions[n * 6:n * 6 + t])
                    #print([normalize_from_df(x, validation_set.loc[:, config["target_variable"]]) for x in
                    # predictions[n * 6:n * 6 + t]])
                    x[0, config["time_steps"] - t:, :] =\
                        [normalize_from_df(x, variables)
                                for x in predictions[n * skip:n * skip + t]]

                prediction = model.predict(x)
                predictions.append(prediction[0])
                var_prediction.append(prediction[0][config["target_variable"].index("target_Power average [kW]")])
                truth_values.append(validation_set.loc[n * skip + config["time_steps"] + t, "target_Power average [kW]"])
            #print(predictions)
            #print(truth_values)
            #print(validation_set.loc[34:40])
            #exit(0)
        #print(truth_values)
        #print(var_prediction)
        error = mean_squared_error(truth_values, var_prediction)
        print('Test MSE: %.3f' % error)
        mae = mean_absolute_error(truth_values, var_prediction)
        print('Test MAE: %.3f' % mae)


#if t == 35:
#    print(x[0, :, config["target_variable"].index("target_Power average [kW]")])
#    serie = validation_set.loc[:, "target_Power average [kW]"]
#    # print(predictions[n * 6:n * 6 + t][config["target_variable"].index("target_Power average [kW]")])
#    print([(x[config["target_variable"].index("target_Power average [kW]")] - serie.mean()) / serie.std() for x in
#           predictions[n * 6:n * 6 + t]])