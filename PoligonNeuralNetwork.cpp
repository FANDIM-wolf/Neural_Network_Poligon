#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
class Neuron {
public:
    Neuron(int numInputs) {
        weights.resize(numInputs);
        for (int i = 0; i < numInputs; i++) {
            weights[i] = ((double)rand() / (RAND_MAX));
        }
        output = 0;
    }

    void activate(std::vector<double>& inputs) {
        double sum = 0;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights[i] * inputs[i];
        }
        output = 1 / (1 + exp(-sum));
    }

    double output;
    std::vector<double> weights;
};

class Layer {
public:
    Layer(int numNeurons, int numInputsPerNeuron) {
        for (int i = 0; i < numNeurons; i++) {
            neurons.push_back(Neuron(numInputsPerNeuron));
        }
    }

    void activate(std::vector<double>& inputs) {
        for (int i = 0; i < neurons.size(); i++) {
            neurons[i].activate(inputs);
        }
    }

    std::vector<Neuron> neurons;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<int> topology) {
        for (int i = 0; i < topology.size() - 1; i++) {
            layers.push_back(Layer(topology[i + 1], topology[i]));
        }
    }

    std::vector<double> feedForward(std::vector<double>& inputs) {
        std::vector<double> outputs;
        layers[0].activate(inputs);
        outputs = std::vector<double>(layers[0].neurons.size());
        for (int i = 0; i < layers[0].neurons.size(); i++) {
            outputs[i] = layers[0].neurons[i].output;
        }
        for (int i = 1; i < layers.size(); i++) {
            layers[i].activate(outputs);
            outputs = std::vector<double>(layers[i].neurons.size());
            for (int j = 0; j < layers[i].neurons.size(); j++) {
                outputs[j] = layers[i].neurons[j].output;
            }
        }
        return outputs;
    }

    void saveWeights(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i].neurons.size(); j++) {
                    for (int k = 0; k < layers[i].neurons[j].weights.size(); k++) {
                        file << layers[i].neurons[j].weights[k];
                        if (k < layers[i].neurons[j].weights.size() - 1) {
                            file << ",";
                        }
                    }
                    file << "\n";
                }
            }
            file.close();
        }
        else {
            std::cout << "Error: could not open file for writing." << std::endl;
        }
    }

    void loadWeights(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            for (int i = 0; i < layers.size(); i++) {
                for (int j = 0; j < layers[i].neurons.size(); j++) {
                    if (file.good()) {
                        std::string line;
                        if (getline(file, line)) {
                            std::stringstream ss(line);
                            for (int k = 0; k < layers[i].neurons[j].weights.size(); k++) {
                                double weight;
                                if (ss.good()) {
                                    std::string weight_str;
                                    getline(ss, weight_str, ',');
                                    weight = std::stod(weight_str);
                                    layers[i].neurons[j].weights[k] = weight;
                                }
                                else {
                                    std::cout << "Error: could not read weight from file." << std::endl;
                                    break;
                                }
                            }
                        }
                        else {
                            std::cout << "Error: could not read line from file." << std::endl;
                            break;
                        }
                    }
                    else {
                        std::cout << "Error: end of file reached prematurely." << std::endl;
                        break;
                    }
                }
            }
            file.close();
        }
        else {
            std::cout << "Error: could not open file for reading." << std::endl;
        }
    }


    std::vector<Layer> layers;

};

int main() {
    // create a neural network with topology {2, 4, 1}
    NeuralNetwork nn({ 2, 4, 1 });

    // train the neural network here...

    // save the weights to a file
    nn.saveWeights("weights.csv");

    // create a new neural network with the same topology
    NeuralNetwork nn2({ 2, 4, 1 });

    // load the weights from the file
    nn2.loadWeights("weights.csv");

    // create input data
    std::vector<double> input = { 2.0, 0.5 , 0.3 };

    // run feedforward algorithm on the loaded network with the input data
    std::vector<double> output = nn2.feedForward(input);

    // print the output as the result
    std::cout << "Result: " << output[0] << std::endl;


    return 0;
}