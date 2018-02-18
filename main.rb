require_relative './neural_network'

input = N[[0.66666667, 1.0, 0.89, 0.45], [0.33333333, 0.55555556, 0.76, 0.81], [1.0, 0.66666667, 0.47, 0.68]]
output = N[[0.92], [0.86], [0.89]]
hidden_layer_size = 4
threshold = 0.0001
NeuralNetwork.build(input, output, hidden_layer_size, threshold)
