#include <stdio.h>
#include <time.h>
#include "cnn.h"

int main() {
    srand(time(NULL));

    Dataset mnist = MNIST("DATASETS/mnist/train-images",
                          "DATASETS/mnist/train-labels",
                          "DATASETS/mnist/test-images",
                          "DATASETS/mnist/test-labels");

    // Initialize neural network
    unsigned int nnShape[] = {784, 16, 16, 10};
    NeuralNet* nn = newNeuralNet(4, nnShape, MeanSquaredError);
    randInit(nn);
    nn->train = true;

    train(nn, SGD, mnist, 10, 1, 0.1, printEpochCallback);

    for (size_t i = 0; i < 10; i++) {
        printf("\n");
        Datum datum = mnist.getTestElement(i);
        printMnistDatum(datum);

        copyTensor(datum.x, nn->input);
        forwardPass(nn);

        printf("Network prediction: %zu\n", argmax(nn->output));
        printf("Confidence: %f%%\n", 100*nn->output->data[argmax(nn->output)]);
        printf("Raw output: ");
        printTensor(nn->output);
    }

    printf("\nSaving neural network...\n");
    saveNeuralNet(nn, "mnist_demo_save.dat");
    printf("Neural network saved\n");
    freeNeuralNet(nn);
}

