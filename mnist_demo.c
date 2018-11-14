#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "neuralnet.h"
#include "loss_functions.h"
#include "mnist_dataset.h"
#include "optimizer.h"
#include "trainer.h"

int main() {
    srand(time(NULL));

    Dataset mnist = MNIST("DATASETS/mnist/train-images",
                          "DATASETS/mnist/train-labels",
                          "DATASETS/mnist/test-images",
                          "DATASETS/mnist/test-labels");

    mnist.trainElements = 10000;
    mnist.testElements = 50;

    // Initialize neural network
    unsigned int nnShape[] = {784, 32, 10};
    NeuralNet* nn = newNeuralNet(3, nnShape, MeanSquaredError);
    randInit(nn);
    nn->train = true;

    train(nn, SGD, mnist, 500, 50, 0.1, printEpochCallback);
}

