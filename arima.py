import sys
import matplotlib.pyplot as plt
from preprocessing import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = read_file(sys.argv[1])
df = clean_data(df)
#df = arrange_data(df)

#BE CAREFUL PREPROCESSING NOT HANDLED
df = df["target_variable"]

training_set, validation_set, test_set = separate_set_seq(df)

history = [x for x in training_set["target_variable"]]
test_set = test_set["target_variable"]
validation_set = validation_set["target_variable"]

model = SARIMAX(history, order=(18,1,9))
model_fit = model.fit(disp=0)

real_model = SARIMAX(validation_set, order=(18,1,9))
res = real_model.filter(model_fit.params)

predictions = []
observations = []
for n in range((len(validation_set) - 36) // 6):
	output = res.get_prediction(start=n*6, dynamic=0, end=n*6+35).predicted_mean
	for t in range(36):
		yhat = output[n*6 + t]
		predictions.append(yhat)
		obs = validation_set[n*6 + t]
		observations.append(obs)
		#print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(observations, predictions)
print('Test MSE: %.3f' % error)
mae = mean_absolute_error(observations, predictions)
print('Test MAE: %.3f' % mae)
# plot