from base_model import *
from RandomBatchGenerator import RandomBatchGenerator
import keras.backend.tensorflow_backend as K
import tensorflow as tf

def custom_mean_squared_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    diff = K.abs(y_true - y_pred)
    angle_diff = K.minimum(diff[:, :, 6:], 360 - diff[:, :, 6:])
    error = tf.concat([diff[:, :, :6], angle_diff], axis=-1)

    return K.mean(K.square(error), axis=-1)

class PaddingModel(base_model):

    def __init__(self, config, model=None):
        self.config = config
        if model is None:
            input_shape = (config["time_steps"] + config["forecast_steps"], len(config["attr"]))

            self.model = Sequential()
            # model.add(Dense(100, kernel_regularizer=config["regularizer"], input_shape=input_shape))
            # model.add(Activation("tanh"))
            self.model.add(
                LSTM(config["hidden_size"], unroll=True, return_sequences=True, kernel_regularizer=config["regularizer"],
                     input_shape=input_shape))
            self.model.add(
                LSTM(config["hidden_size"], unroll=True, return_sequences=True, kernel_regularizer=config["regularizer"]))
            self.model.add(Lambda(lambda x: x[:, -config["forecast_steps"]:, :]))
            self.model.add(Dense(len(config["target_variable"]), kernel_regularizer=config["regularizer"], activation="relu"))

            # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

            self.model.compile(loss=custom_mean_squared_loss, optimizer='adam', metrics=["mean_absolute_error", "mean_squared_error"])
            print(self.model.summary())
        else:
            self.model = load_model(model)

    def train(self, training_set, validation_set, plotLoss=True):
        model = self.model
        config = self.config

        training_generator = RandomBatchGenerator(training_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                                  config["target_variable"], config["batch_size"], fill=0.0)
        validation_generator = PandasBatchGenerator(validation_set, config["time_steps"], config["forecast_steps"], config["attr"],
                                                    config["target_variable"], config["batch_size"], config["skip_steps"], fill=0.0)

        # inp, out =  next(training_generator.generate())
        history = model.fit_generator(training_generator.generate(), len(training_set) // (
                    (config["batch_size"] * config["skip_steps"]) + config["time_steps"] + config["forecast_steps"]),
                                      config["num_epochs"], validation_data=validation_generator.generate(),
                                      validation_steps=len(validation_set) // (
                                                  (config["batch_size"] * config["skip_steps"]) + config["time_steps"] +
                                                  config["forecast_steps"]), shuffle="batch",
                                      )
        print("nb inputs skipped = %d" % (len(training_generator.idx_errors)))

        if plotLoss:
            self.plot_infos(history)