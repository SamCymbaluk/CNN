#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "neuralnet.h"
#include "loss_functions.h"
#include "dataset.h"
#include "optimizer.h"
#include "trainer.h"


Datum xorElements[4];

void setupXorElements() {
    unsigned int shape[] = {2, 1};

    for (int i = 0; i < 4; i++) {
        float input1 = i % 2;
        float input2 = (i >> 1) % 2;
        int output = input1 != input2;

        Tensor* x = newTensor(2, shape);
        Tensor* y = newTensor(2, shape);

        x->data[0] = input1;
        x->data[1] = input2;

        y->data[output] = 1.0;

        xorElements[i] = (Datum) {
                .x = x,
                .y = y
        };
    }
}

Datum getXorElement(size_t index) {
    return xorElements[index % 4];
}

void xorShuffle() {}


int main() {
    srand(time(NULL));

    setupXorElements();

    // Initialize neural network
    unsigned int nnShape[] = {2, 16, 16, 2};
    NeuralNet* nn = newNeuralNet(4, nnShape, MeanSquaredError);
    randInit(nn);
    nn->train = true;

    Dataset xorDataset = (Dataset) {
        .trainElements = 5000,
        .getTrainElement = getXorElement,
        .testElements = 4,
        .getTestElement = getXorElement,
        .shuffle = xorShuffle,
    };

    train(nn, SGD, xorDataset, 20, 20, 0.1, printEpochCallback);
}

