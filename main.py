import numpy as np
import neuralnetwork as nn

with np.load('data/mnist.npz') as data: 
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

layer_sizes = (784, 5, 10)

net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(test_images, test_labels)
net.calculate_average_cost(test_images, test_labels)