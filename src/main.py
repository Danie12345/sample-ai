from turtle import update
import numpy as np
import pandas as pd
from functions import gen_data, get_weighted_sum, sigmoid, cross_entropy, update_weights, update_bias

bias = .5
l_rate = .05

data, weights = gen_data(4, 3)

for i in range(len(data)):
	feature = data.loc[i][:-1]
	target = data.loc[i][-1]
	w_sum = get_weighted_sum(feature, weights, bias)
	prediction = sigmoid(w_sum)
	loss = cross_entropy(target, prediction)
	weights = update_weights(weights, l_rate, target, prediction, feature)
	bias = update_bias(bias, l_rate, target, prediction)
	print (weights, bias)