# WindPowerForecasting
Forecast wind power and electricity production of wind turbines

This repository contains several models and allows to easily modify several hyper parameters.

### KerasBatchGenerator
Classic batch generator to configure, which provide inputs and teaching signals

### PandasBatchGenerator
Extract any sequence randomly in the training set and feeds it to the network

### PaddedModel
Use this model for a many to many architecture such as on this figure. This model has to be fed *time_steps* (via the config) samples as input and will predict *forecast_steps* steps in the future.

### StatefulModel
Model should be trained with *forecast_steps*=1. The model uses the Stateful parameter of keras, and predictions of the model are used as input for next steps, such as in the figure below.

### PersistenceModel
The oversimplyfied persistence model used as a benchmark algorithm.

### Arima
File containing an ARIMA model to configure and test for benchmarking.
