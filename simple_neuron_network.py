import random
import math

class Neuron:
    def __init__(self, num_weights):
        self.weights = [random.uniform(0, 1) for _ in range(num_weights)]
        self.output = 0.0

class NeuralLayer:
    def __init__(self, num_neurons, num_weights):
        self.neurons = [Neuron(num_weights) for _ in range(num_neurons)]

class NeuralNetwork:
    def __init__(self, topology):
        self.layers = []
        for i in range(1, len(topology)):
            self.layers.append(NeuralLayer(topology[i], topology[i-1]))

    def feed_forward(self, inputs):
        for i in range(len(inputs)):
            self.layers[0].neurons[i].output = inputs[i]
        
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                sum_weights_outputs = sum(weight * prev_neuron.output for weight, prev_neuron in zip(neuron.weights, self.layers[i-1].neurons))
                neuron.output = self.sigmoid(sum_weights_outputs)

        return [neuron.output for neuron in self.layers[-1].neurons]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def save_weights(self, filename):
        with open(filename, 'w') as file:
            for layer in self.layers:
                for neuron in layer.neurons:
                    weights_str = ','.join(str(weight) for weight in neuron.weights)
                    file.write(weights_str + '\n')

    def load_weights(self, filename):
        with open(filename, 'r') as file:
            for layer in self.layers:
                for neuron in layer.neurons:
                    line = file.readline().strip()
                    weights = [float(weight) for weight in line.split(',')]
                    neuron.weights = weights

# Example usage
nn = NeuralNetwork([2, 4, 1])  # create a neural network with topology [2, 4, 1]

# Train the neural network here...

nn.save_weights('weights.csv')  # save the weights to a file

nn2 = NeuralNetwork([2, 4, 1])  # create a new neural network with the same topology

nn2.load_weights('weights.csv')  # load the weights from the file

inputs = [1.0, 0.5]  # create input data

output = nn2.feed_forward(inputs)  # run feedforward algorithm on the loaded network with the input data

print('Result:', output)  # print the output as the result
