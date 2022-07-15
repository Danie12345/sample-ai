import numpy as np
import pandas as pd

rg = np.random.default_rng()

def gen_data(n_features, n_values):
	features = rg.random((n_features, n_values))
	weights = rg.random((1, n_values))[0]
	targets = np.random.choice([0, 1], n_features)
	data = pd.DataFrame(features, columns = ["x0", "x1", "x2"])
	data["targets"] = targets
	return data, weights

def get_weighted_sum(feature, weights, bias):
	return np.dot(feature, weights) + bias

def sigmoid(w_sum):
	return 1/(1 + np.exp(-w_sum))

def cross_entropy(target, prediction):
	return -(target * np.log10(prediction) + (1 + target) * np.log10(1 - prediction))