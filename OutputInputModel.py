from base_model import *
from preprocessing import normalize_from_df
import cProfile

class OutputInputModel(base_model):

    def get_stateful_model(self, config):
        input_shape = (1, None, len(config["attr"]))
        model = Sequential()
        # model.add(Dense(100, kernel_regularizer=config["regularizer"], input_shape=input_shape))
        # model.add(Activation("tanh"))
        model.add(
          LSTM(config["hidden_size"], return_sequences=True, kernel_regularizer=config["regularizer"],
                batch_input_shape=input_shape, stateful=True))

        model.add(
            LSTM(config["hidden_size"], return_sequences=False, kernel_regularizer=config["regularizer"], stateful=True))
        model.add(Dense(len(config["target_variable"]), kernel_regularizer=config["regularizer"]))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_absolute_error"])

        return model

    def __init__(self, config, model=None):
        self.config = config
        input_shape = (config["batch_size"], None, len(config["attr"]))

        if model is None:
            self.model = Sequential()
            # model.add(Dense(100, kernel_regularizer=config["regularizer"], input_shape=input_shape))
            # model.add(Activation("tanh"))
            self.model.add(
                LSTM(config["hidden_size"], return_sequences=True, kernel_regularizer=config["regularizer"],
                     batch_input_shape=input_shape))

            self.model.add(
                LSTM(config["hidden_size"], return_sequences=False, kernel_regularizer=config["regularizer"],
                     ))
            self.model.add(Dense(len(config["target_variable"]), kernel_regularizer=config["regularizer"]))

            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=["mean_absolute_error"])
            self.callbacks = []
            #self.callbacks.append(LambdaCallback(on_epoch_end=lambda batch, logs: self.model.reset_states()))

            print(self.model.summary())
        else:
            self.model = load_model(model)


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
                                                  config["forecast_steps"]), shuffle=False, callbacks=self.callbacks
                                      )
        print("nb inputs skipped = %d" % (len(training_generator.idx_errors)))

        if plotLoss:
            self.plot_infos(history)

    def output_as_input_testing(self, validation_set, variable_to_test, stateful=True):
        if stateful:
            model = self.get_stateful_model(self.config)
            model.set_weights(self.model.get_weights())
        else:
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
                    try:
                        x[0, config["time_steps"] - t:, :] =\
                            [normalize_from_df(x, variables)
                                    for x in predictions[n * skip:n * skip + t]]
                    except Exception as e:
                        print(e)
                        print("n = %d t = %d" % (n, t))
                        print(predictions[n * skip:n * skip + t])

                prediction = model.predict(x)
                #exit(0)
                predictions.append(prediction[0])
                var_prediction.append(prediction[0][config["target_variable"].index(variable_to_test)])
                truth_values.append(validation_set.loc[n * skip + config["time_steps"] + t, variable_to_test])
            #print(predictions)
            #print(truth_values)
            #print(validation_set.loc[34:40])
            #exit(0)
        #print(truth_values)
        #print(var_prediction)
        print(var_prediction[0:36])
        print(truth_values[0:36])
        print("")
        print(var_prediction[36:72])
        print(truth_values[36:72])
        print("")
        print(var_prediction[72:108])
        print(truth_values[72:108])
        print("")
        error = mean_squared_error(truth_values, var_prediction)
        print('Test MSE: %.3f' % error)
        mae = mean_absolute_error(truth_values, var_prediction)
        print('Test MAE: %.3f' % mae)
        order = abs((mae / validation_set[variable_to_test].mean()) * 100)
        print("Error of order : %d%%" % order)


#if t == 35:
#    print(x[0, :, config["target_variable"].index("target_Power average [kW]")])
#    serie = validation_set.loc[:, "target_Power average [kW]"]
#    # print(predictions[n * 6:n * 6 + t][config["target_variable"].index("target_Power average [kW]")])
#    print([(x[config["target_variable"].index("target_Power average [kW]")] - serie.mean()) / serie.std() for x in
#           predictions[n * 6:n * 6 + t]])