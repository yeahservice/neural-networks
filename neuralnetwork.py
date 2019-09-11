import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in weight_shapes]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    def feedforward (self, a):
        for w, b in zip(self.weights, self.biases):
            a = NeuralNetwork.activation(np.matmul(w, a) + b)
        return a

    def train_sgd(self, training_data, training_labels, epochs, batch_size, learning_rate):
        """
        Train the neural network using stochastic gradient decent
        """

    def print_accuracy(self, images, labels):
        predictions = self.feedforward(images)
        num_correct = sum([np.argmax(p) == np.argmax(l) for p, l in zip(predictions, labels)])
        print("{0}/{1} accuracy: {2}%".format(num_correct, len(images), (num_correct / len(images)) * 100))

    def calculate_average_cost(self, images, labels):
        predictions = self.feedforward(images)
        average_cost = sum([NeuralNetwork.cost_function(p, l) for p, l in zip(predictions, labels)]) / len(images)
        print("average cost: {0}".format(average_cost))

    @staticmethod
    def activation(x):
        """
        sigmoid
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def cost_function(output, y):
        """
        squared error
        """
        return sum([(a - b) ** 2 for a, b in zip(output, y)])[0]

    @staticmethod
    def cost_function_derivation(output, y):
        return 2 * (output - y)
