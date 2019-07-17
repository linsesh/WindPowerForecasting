from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Lambda, LSTM
import matplotlib.pyplot as plt
from batch_generator import PandasBatchGenerator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

    def test_one_variable_mutivariate_model(self, validation_set):
        model = self.model
        config = self.config
        predictions = []
        observations = []
        for n in range((len(validation_set) - 36) // 6):
            x = np.full((1, config["forecast_steps"] + config["time_steps"], len(config["attr"])), 0.0)
            x[0,:config["time_steps"],:] = validation_set.loc[n * 6:n * 6 + 35, config["attr"]]
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

    def save(self, path):
        self.model.save(path)