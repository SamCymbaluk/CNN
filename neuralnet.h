#ifndef PROJECT_NEURALNET_H
#define PROJECT_NEURALNET_H

#include "tensor.h"
#include "loss_functions.h"

typedef struct NeuralNet NeuralNet;

struct NeuralNet {
    unsigned int depth; // Number of layers
    unsigned int* shape; // Array of length depth

    bool train;

    Tensor* input; // Input layer
    Tensor* output; // Output layer
    Tensor** layers; // Array of pointers to Tensors of length depth
    Tensor** zs; // Array of pointers to Tensors that correspond to layers without the activations applied
    Tensor** weights; // Array of pointers to tensors of length depth - 1 that stores the connection weights
    Tensor** biases; // Array of pointers to tensors of length depth - 1 that stores the biases for layer 2 onwards

    LossFunction* lossFunction;
};

NeuralNet* newNeuralNet(unsigned int depth, unsigned int* shape, LossFunction* lossFunction);

void freeNeuralNet(NeuralNet* nn);

void randInit(NeuralNet* nn);

void forwardPass(NeuralNet* nn);

Tensor*** backProp(NeuralNet *nn, Tensor* yTrue);

void batchTrain(NeuralNet *nn, Tensor** xy[], int batchSize, float lr);

void applyBackProp(NeuralNet* nn, Tensor*** wb, float lr);

void freeBackProp(NeuralNet* nn, Tensor*** wb);

#endif //PROJECT_NEURALNET_H
