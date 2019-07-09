import sys
import matplotlib.pyplot as plt
from preprocessing import *
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


df = read_file(sys.argv[1])
df = clean_data(df)
#df = arrange_data(df)

#BE CAREFUL PREPROCESSING NOT HANDLED
df = df["target_variable"]

training_set, validation_set, test_set = separate_set_seq(df)
history = [x for x in training_set["target_variable"]]
predictions = list()
for t in range(len(test_set)):
	model = ARIMA(history, order=(18,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test_set[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test_set, predictions)
print('Test MSE: %.3f' % error)
# plot