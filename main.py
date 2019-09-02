import numpy as np
import neuralnetwork as nn

with np.load('data/mnist.npz') as data: 
    training_images = data['training_images']
    training_labels = data['training_labels']

# plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
# plt.show()

layer_sizes = (784,5,10) # tuple

net = nn.NeuralNetwork(layer_sizes)
net.print_accuracy(training_images, training_labels)