from turtle import update
import numpy as np
import pandas as pd
from functions import gen_data, get_weighted_sum, sigmoid, cross_entropy, update_weights, update_bias

bias = .5
l_rate = .001
epochs = 20000
epoch_loss = []
data, weights = gen_data(50, 3)

def train_model(data, weights, bias, l_rate=l_rate, epochs=epochs):
	for e in range(epochs):
		individual_loss = []
		for i in range(len(data)):
			feature = data.loc[i][:-1]
			target = data.loc[i][-1]
			w_sum = get_weighted_sum(feature, weights, bias)
			prediction = sigmoid(w_sum)
			loss = cross_entropy(target, prediction)
			individual_loss.append(loss)
			weights = update_weights(weights, l_rate, target, prediction, feature)
			bias = update_bias(bias, l_rate, target, prediction)
		avg_loss = sum(individual_loss) / len(individual_loss)
		epoch_loss.append(avg_loss)

train_model(data, weights, bias, l_rate, epochs)

df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind='line', grid=True).get_figure()
df_plot.savefig('src/assets/training/training_loss.png')