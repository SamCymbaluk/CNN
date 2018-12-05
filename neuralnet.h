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

    LossFunction lossFunction;
};

typedef Tensor** NNWeightsBiases;

NeuralNet* newNeuralNet(unsigned int depth, unsigned int* shape, LossFunction lossFunction);

void freeNeuralNet(NeuralNet* nn);

void randInit(NeuralNet* nn);

/**
 * Save Neural Network weights and biases to a binary file for loading later
 * File schema:
 * [offset] | [type]  | [desc]
 * 0000       uint32    Network depth
 * 0004       uint32    Shape of layer 1
 * 0008       uint32    Shape of layer 2
 * ...
 * 00xx       uint32    Shape of layer n - 1
 * 0xxx       float32   weights[0]
 * ...
 * 0xxx       float32   biases[0]
 * ...
 * xxxx       float32   weights[n - 1]
 * ...
 * xxxx       float32   biases[n - 1]
 * @param nn
 * @param fileName
 */
void saveNeuralNet(NeuralNet* nn, char* fileName);

void loadNeuralNet(NeuralNet* nn, char* fileName);

void forwardPass(NeuralNet* nn);

NNWeightsBiases* newWeightBiasUpdate(NeuralNet* nn);

void scaleWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* wb, float scalar);

void copyWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* src, NNWeightsBiases* dest);

void addWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* a, NNWeightsBiases* b, NNWeightsBiases* c);

void freeWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* wb);

void backProp(NeuralNet *nn, NNWeightsBiases* wb, Tensor* yTrue);

void applyBackProp(NeuralNet* nn, NNWeightsBiases* wb, float lr);

#endif //PROJECT_NEURALNET_H
