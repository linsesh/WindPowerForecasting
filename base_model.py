from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Lambda, LSTM
import matplotlib.pyplot as plt
from batch_generator import PandasBatchGenerator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
from visualization import plot_predicted_vs_truth
from random import randrange

class base_model:

    def train(self):
        pass

    def test(self):
        pass

    def get_model(self):
        return self.model

    def plot_infos(self, history):
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
        plt.close()

    def keras_prediction_example(self, test_set, config, model, n=200):
        test_generator = PandasBatchGenerator(test_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                              config["target_variable"], 1, config["skip_steps"])

        print("example of prediction :")
        for i in range(n):
            # print(test_set[test_generator.current_idx:test_generator.current_idx + test_generator.num_steps + test_generator.num_padding])
            inp, out = next(test_generator.generate())
            y = model.predict(inp, verbose=1)
            print(y)
            print(out)

    def keras_batch_testing(self, test_set, model, config):
        test_generator = PandasBatchGenerator(test_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                              config["target_variable"], 1, config["skip_steps"])

        ev = model.evaluate_generator(test_generator.generate(),
                                      len(test_set) // (
                                                  (config["batch_size"] * config["skip_steps"]) + config["time_steps"] +
                                                  config["forecast_steps"]),
                                      verbose=1)
        print("error on test set: [%s]" % ', '.join(map(str, ev)))

    def test_variables_mutivariate_model(self, validation_set, variables_to_test, oresults, olabels, plotName=None):
        model = self.model
        config = self.config
        skip = config["skip_steps"]
        predictions = {}
        observations = {}
        for x in variables_to_test:
            predictions[x] = []
            observations[x] = []

        for n in range((len(validation_set) - config["forecast_steps"] - config["time_steps"]) // skip):
            x = np.full((1, config["forecast_steps"] + config["time_steps"], len(config["attr"])), 0.0)
            x[0,:config["time_steps"],:] = validation_set.loc[n * skip:n * skip + config["time_steps"] - 1, config["attr"]]
            output = model.predict(x)
            for t in range(config["forecast_steps"]):
                for variable in variables_to_test:
                    yhat = output[0][t][config["target_variable"].index(variable)]
                    predictions[variable].append(yhat)
                    obs = validation_set.loc[n * skip + t, variable]
                    observations[variable].append(obs)
                #print('predicted=%f, expected=%f' % (yhat, obs))
        ret = []
        for x in variables_to_test:
            mse = mean_squared_error(observations[x], predictions[x])
            print('Test MSE: %.3f' % mse)
            mae = mean_absolute_error(observations[x], predictions[x])
            print('Test MAE: %.3f' % mae)
            order = abs((mae / validation_set[x].mean()) * 100)
            print("Error of order : %d%%" % order)
            idx = randrange(len(observations[x]))
            idx = idx - 72 if idx + 72 > len(observations[x]) else idx
            ret.append((mae, mse))
            oresults.append(predictions[x][17434:17434 + 72])
            olabels.append("LSTM")
            plot_predicted_vs_truth(oresults, olabels, observations[x][17434:17434+72], validation_set[x].min(), validation_set[x].max(), plotName)
        return ret


    def save(self, path):
        self.model.save(path)