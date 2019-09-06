import sys
import matplotlib.pyplot as plt
from preprocessing import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import get_config
from visualization import plot_predicted_vs_truth
from random import randrange



def fit_arima(training_set, validation_set):
	training_set = clean_data(training_set, [])
	validation_set = clean_data(validation_set, [])
	training_set.reset_index(drop=True, inplace=True)
	validation_set.reset_index(drop=True, inplace=True)

	target_variable = "Wind average [m/s]"
	history = [x for x in training_set[target_variable]]
	#test_set = test_set[target_variable]
	validation_set = validation_set[target_variable]

	model = SARIMAX(history, order=(9, 1, 1))
	model_fit = model.fit(disp=0)

	real_model = SARIMAX(validation_set, order=(9, 1, 1))
	res = real_model.filter(model_fit.params)

	config = get_config(None, None)

	predictions = []
	observations = []
	for n in range((len(validation_set) - config["forecast_steps"]) // config["skip_steps"]):
		output = res.get_prediction(start=n * config["skip_steps"], dynamic=0,
									end=n * config["skip_steps"] + config[
										"forecast_steps"] - 1).predicted_mean.to_numpy()

		for t in range(config["forecast_steps"]):
			yhat = output[t]
			predictions.append(yhat)
			obs = validation_set[n * config["skip_steps"] + t]
			observations.append(obs)
	# print('predicted=%f, expected=%f' % (yhat, obs))

	error = mean_squared_error(observations, predictions)
	print('Test MSE: %.3f' % error)
	mae = mean_absolute_error(observations, predictions)
	print('Test MAE: %.3f' % mae)
	order = abs((mae / validation_set.mean()) * 100)
	print("Error of order : %d%%" % order)
	idx = randrange(len(observations))
	#plot_predicted_vs_truth(predictions[idx:idx + 72], observations[idx:idx + 72], validation_set.min(),
							#validation_set.max())
	return predictions[17434:17434+72]


if __name__ == "__main__":
	df = read_file(sys.argv[1])

	training_set, validation_set, test_set = separate_set_seq(df, 80, 19, 1)
	fit_arima(training_set, validation_set)


# plot