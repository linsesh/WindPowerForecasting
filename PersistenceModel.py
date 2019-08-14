from base_model import *

class PersistenceModel(base_model):
    def __init__(self, config):
        self.config = config

    def test_variables_mutivariate_model(self, validation_set, variables_to_test, plotName=None):
        config = self.config
        skip = config["skip_steps"]
        predictions = {}
        observations = {}
        for x in variables_to_test:
            predictions[x] = []
            observations[x] = []

        for n in range((len(validation_set) - config["forecast_steps"] - config["time_steps"]) // skip):
            for var in variables_to_test:
                predictions[var].extend([validation_set.loc[n * skip + config["time_steps"] - 1, var]] * config["forecast_steps"]) #one single prediction for all the time steps
                observations[var].extend(validation_set.loc[n * skip + config["time_steps"]:n * skip + config["time_steps"] + config["forecast_steps"] - 1, var].values.tolist())
        for x in variables_to_test:
            error = mean_squared_error(observations[x], predictions[x])
            print('Test MSE: %.3f' % error)
            mae = mean_absolute_error(observations[x], predictions[x])
            print('Test MAE: %.3f' % mae)
            order = abs((mae / validation_set[x].mean()) * 100)
            print("Error of order : %d%%" % order)
            idx = randrange(len(observations[x]))
            idx = idx - 72 if idx + 72 > len(observations[x]) else idx
            r = idx % 72
            #plot_predicted_vs_truth(predictions[x][idx-r:idx-r+72], observations[x][idx:idx+72], validation_set[x].min(), validation_set[x].max())

