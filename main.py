import numpy as np
import neuralnetwork as nn

with np.load('data/mnist.npz') as data: 
    training_images = data['training_images']
    training_labels = data['training_labels']

# plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
# plt.show()

training_set_size = 500

training_set_images = training_images[:training_set_size]
training_set_labels = training_labels[:training_set_size]

test_set_images = training_images[training_set_size:]
test_set_labels = training_labels[training_set_size:]

layer_sizes = (784, 5, 10)

net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(test_set_images, test_set_labels)
net.calculate_average_cost(test_set_images, test_set_labels)