
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
        for i in range(min(len(inputs), len(self.layers[0].neurons))):
            self.layers[0].neurons[i].output = inputs[i]

        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                neuron = self.layers[i].neurons[j]
                sum_weights_outputs = sum(weight * prev_neuron.output for weight, prev_neuron in zip(neuron.weights, self.layers[i-1].neurons))
                neuron.output = sum_weights_outputs

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


nn2 = NeuralNetwork([4, 4, 10 , 10 , 10 , 10 ,2])  # create a new neural network with the same topology


inputs = [4,2,3,33]  # create input data

output = nn2.feed_forward(inputs)  # run feedforward algorithm on the loaded network with the input data

print('Result:', output[0])  # print the output as the result



