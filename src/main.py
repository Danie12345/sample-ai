import numpy as np
import pandas as pd
from functions import gen_data, get_weighted_sum, sigmoid, cross_entropy

bias = .5

data, weights = gen_data(4, 3)

for i in range(len(data)):
	feature = data.loc[i][:-1]
	target = data.loc[i][-1]
	w_sum = get_weighted_sum(feature, weights, bias)
	prediction = sigmoid(w_sum)
	loss = cross_entropy(target, prediction)
	print(loss)