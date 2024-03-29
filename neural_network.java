import java.util.ArrayList;
import java.util.List;
import java.util.Random;

class Neuron {
    private List<Double> weights;
    private double output;

    public Neuron(int numWeights) {
        weights = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < numWeights; i++) {
            weights.add(random.nextDouble());
        }
        output = 0.0;
    }
}

class NeuralLayer {
    private List<Neuron> neurons;

    public NeuralLayer(int numNeurons, int numWeights) {
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            neurons.add(new Neuron(numWeights));
        }
    }
}

class NeuralNetwork {
    private List<NeuralLayer> layers;

    public NeuralNetwork(int[] topology) {
        layers = new ArrayList<>();
        for (int i = 1; i < topology.length; i++) {
            layers.add(new NeuralLayer(topology[i], topology[i - 1]));
        }
    }

    public double[] feedForward(double[] inputs) {
        for (int i = 0; i < inputs.length; i++) {
            layers.get(0).neurons.get(i).output = inputs[i];
        }

        for (int i = 1; i < layers.size(); i++) {
            for (int j = 0; j < layers.get(i).neurons.size(); j++) {
                Neuron neuron = layers.get(i).neurons.get(j);
                double sumWeightsOutputs = 0.0;
                for (int k = 0; k < neuron.weights.size(); k++) {
                    sumWeightsOutputs += neuron.weights.get(k) * layers.get(i - 1).neurons.get(k).output;
                }
                neuron.output = sigmoid(sumWeightsOutputs);
            }
        }

        double[] output = new double[layers.get(layers.size() - 1).neurons.size()];
        for (int i = 0; i < output.length; i++) {
            output[i] = layers.get(layers.size() - 1).neurons.get(i).output;
        }
        return output;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}

public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 4, 1}); // create a neural network with topology [2, 4, 1]

        // Train the neural network here...

        double[] inputs = {1.0, 0.5}; // create input data

        double[] output = nn.feedForward(inputs); // run feedforward algorithm on the network with the input data

        System.out.print("Result: ");
        for (double value : output) {
            System.out.print(value + " ");
        }
        System.out.println(); // print the output as the result
    }
}
