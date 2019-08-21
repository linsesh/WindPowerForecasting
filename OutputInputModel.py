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

    def output_as_input_testing(self, validation_set, variables_to_test, stateful=True, forecast_steps=None):
        config = self.config
        config["batch_size"] = 1
        if stateful:
            model = self.get_stateful_model(config)
            model.set_weights(self.model.get_weights())
        else:
            model = self.model

        if forecast_steps is None:
            forecast = config["forecast_steps"]
        else:
            forecast = forecast_steps
        predictions = []
        var_prediction = {}
        truth_values = {}
        for x in variables_to_test:
            var_prediction[x] = []
            truth_values[x] = []
        variables = validation_set[config["target_variable"]]
        skip = config["skip_steps"]
        for n in range((len(validation_set) - (forecast + config["time_steps"])) // skip):
            x = np.zeros((1, config["time_steps"], len(config["attr"])))
            model.reset_states()
            for t in range(forecast):
                if t < config["time_steps"]:
                    x[0, :config["time_steps"] - t, :] = validation_set.loc[n * skip + t:n * skip + config["time_steps"] - 1,
                                                         config["attr"]]
                if t > 0:
                    #print(predictions[n * 6:n * 6 + t])
                    #print([normalize_from_df(x, validation_set.loc[:, config["target_variable"]]) for x in
                    # predictions[n * 6:n * 6 + t]])
                    try:
                        pred = [normalize_from_df(x, variables)
                                    for x in predictions[n * skip + ((t // config["time_steps"]) * t % config["time_steps"]) :n * skip + t]]
                        if t < config["time_steps"]:
                            x[0, config["time_steps"] - t:, :] = pred
                        else:
                            x[0, :, :] = pred

                    except Exception as e:
                        print(e)
                        print("n = %d t = %d" % (n, t))
                        print("hey : ")
                        print([normalize_from_df(x, variables)
                                    for x in predictions[n * skip:n * skip + t]])
                        print(predictions[n * skip:n * skip + t])
                        exit(0)


                prediction = model.predict(x)
                predictions.append(prediction[0])#HEIN ? ptet bon mais bon

                #exit(0)
                for var in variables_to_test:
                    var_prediction[var].append(prediction[0][config["target_variable"].index(var)])
                    truth_values[var].append(validation_set.loc[n * skip + config["time_steps"] + t, var])
            #print(predictions)
            #print(truth_values)
            #print(validation_set.loc[34:40])
            #exit(0)
        #print(truth_values)
        #print(var_prediction)
        ret = []
        for var in variables_to_test:
            mse = mean_squared_error(truth_values[var], var_prediction[var])
            print('Test MSE: %.3f' % mse)
            mae = mean_absolute_error(truth_values[var], var_prediction[var])
            print('Test MAE: %.3f' % mae)
            order = abs((mae / validation_set[var].mean()) * 100)
            print("Error of order : %d%%" % order)
            ret.append((mae, mse))
        return ret


#if t == 35:
#    print(x[0, :, config["target_variable"].index("target_Power average [kW]")])
#    serie = validation_set.loc[:, "target_Power average [kW]"]
#    # print(predictions[n * 6:n * 6 + t][config["target_variable"].index("target_Power average [kW]")])
#    print([(x[config["target_variable"].index("target_Power average [kW]")] - serie.mean()) / serie.std() for x in
#           predictions[n * 6:n * 6 + t]])