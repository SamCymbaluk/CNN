#include <stdlib.h>
#include <stdio.h>

#include "neuralnet.h"

/**
 * Creates a new neural net with a specified number of fully connected layers
 * @param depth Number of layers
 * @param shape Array of length 'depth' that specifies the size of each layer
 * @return
 */
NeuralNet* newNeuralNet(unsigned int depth, unsigned int* shape, LossFunction lossFunction) {
    NeuralNet* nn = malloc(sizeof(NeuralNet));

    nn->depth = depth;
    nn->shape = calloc(depth, sizeof(unsigned int));
    nn->train = false;
    nn->layers = calloc(depth, sizeof(Tensor*));
    nn->zs = calloc(depth, sizeof(Tensor*));
    nn->weights = calloc(depth - 1, sizeof(Tensor*));
    nn->biases = calloc(depth - 1, sizeof(Tensor*));

    for (int i = 0; i < depth; i++) {
        // Initialize shape element
        nn->shape[i] = shape[i];

        // Layers
        unsigned int layerShape[2];
        layerShape[0] = shape[i]; layerShape[1] = 1;
        nn->layers[i] = newTensor(2, layerShape);
        nn->zs[i] = newTensor(2, layerShape);

        // Weights and biases (for layers after the first)
        if (i) {
            // Biases
            nn->biases[i - 1] = newTensor(2, layerShape);

            // Weights
            unsigned int weightShape[2];
            weightShape[0] = shape[i]; weightShape[1] = shape[i - 1];
            nn->weights[i - 1] = newTensor(2, weightShape);
        }
    }

    nn->input = nn->layers[0];
    nn->output = nn->layers[depth - 1];

    nn->lossFunction = lossFunction;

    return nn;
}

void freeNeuralNet(NeuralNet* nn) {
    for (int i = 0; i < nn->depth; i++) {
        freeTensor(nn->layers[i]);
        freeTensor(nn->zs[i]);
        if (i) {
            freeTensor(nn->biases[i - 1]);
            freeTensor(nn->weights[i - 1]);
        }
    }
    free(nn->biases);
    free(nn->weights);
    free(nn->layers);
    free(nn->shape);
    free(nn);
}

void randInit(NeuralNet* nn) {
    for (int n = 0; n < nn->depth - 1; n++) {
        randomize(nn->weights[n], -1.0f, 1.0f);
        randomize(nn->biases[n], -1.0f, 1.0f);
    }
}

void saveNeuralNet(NeuralNet* nn, char* fileName) {
    FILE *file = fopen(fileName, "wb");

    if (file) {

        // Network details
        fwrite(&(nn->depth), 4, 1, file);
        fwrite(nn->shape, 4, nn->depth, file);

        // Weights and biases
        for (unsigned int i = 0; i < nn->depth - 1; i++) {
            fwrite(nn->weights[i]->data, 4, nn->weights[i]->size, file);
            fwrite(nn->biases[i]->data, 4, nn->biases[i]->size, file);
        }

        fclose(file);
    } else {
        fprintf(stderr, "Could not open file for saveNeuralNetwork");
    }
}

void loadNeuralNet(NeuralNet* nn, char* fileName) {
    FILE* file = fopen(fileName, "rb");

    if (file) {

        // Network details
        unsigned int depth;
        fread(&depth, 4, 1, file);
        if (depth != nn->depth) {
            fprintf(stderr, "NeuralNet depth does not match");
            exit(100);
        }
        for (unsigned int i = 0; i < nn->depth; i++) {
            unsigned int shape;
            fread(&shape, 4, 1, file);
            if (shape != nn->shape[i]) {
                fprintf(stderr, "NeuralNet shape does not match");
                exit(100);
            }
        }

        // Weights and biases
        for (unsigned int i = 0; i < nn->depth - 1; i++) {
            fread(nn->weights[i]->data, sizeof(float), nn->weights[i]->size, file);
            fread(nn->biases[i]->data, sizeof(float), nn->biases[i]->size, file);
        }

        fclose(file);
    } else {
        fprintf(stderr, "Could not open file for loadNeuralNetwork");
    }
}

void forwardPass(NeuralNet* nn) {
    // A_n = sigmoid (W_n * A_(n - 1) + B_n)
    for (int n = 1; n < nn->depth; n++) {

        // A_n = W_n * A_(N - 1)
        matmul(nn->weights[n - 1], nn->layers[n - 1], nn->layers[n]);

        // A_n = A_n + B_n
        add(nn->layers[n], nn->biases[n - 1], nn->layers[n]);

        // Store pre-activation values in z
        if (nn->train) copyTensor(nn->layers[n], nn->zs[n]);

        // A_n = sigmoid (A_N)
        if (n == nn->depth - 1) {
            softmax(nn->layers[n]); // Apply softmax instead of sigmoid on final output layer
        } else {
            sigmoid(nn->layers[n]);
        }
    }
}

NNWeightsBiases* newWeightBiasUpdate(NeuralNet* nn) {
    unsigned int layers = nn->depth;

    NNWeightsBiases* wb = calloc(2, sizeof(Tensor**));

    wb[0] = calloc(layers - 1, sizeof(Tensor*));
    wb[1] = calloc(layers - 1, sizeof(Tensor*));

    unsigned int layerShape[2];

    for (int i = 0; i < layers - 1; i++) {
        layerShape[0] = nn->shape[i + 1]; layerShape[1] = 1;
        // Biases
        wb[1][i] = newTensor(2, layerShape);

        // Weights
        unsigned int weightShape[2];
        weightShape[0] = nn->shape[i + 1]; weightShape[1] = nn->shape[i];
        wb[0][i] = newTensor(2, weightShape);
    }

    return wb;
}

void scaleWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* wb, float scalar) {
    for (int i = 0; i < nn->depth - 1; i++) {
        scalarmult(wb[0][i], scalar);
        scalarmult(wb[1][i], scalar);
    }
}

void copyWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* src, NNWeightsBiases* dest) {
    for (int i = 0; i < nn->depth - 1; i++) {
        copyTensor(src[0][i], dest[0][i]);
        copyTensor(src[1][i], dest[1][i]);
    }
}

void addWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* a, NNWeightsBiases* b, NNWeightsBiases* c) {
    for (int i = 0; i < nn->depth - 1; i++) {
        add(a[0][i], b[0][i], c[0][i]);
        add(a[1][i], b[1][i], c[1][i]);
    }
}

void freeWeightBiasUpdate(NeuralNet* nn, NNWeightsBiases* wb) {
    for (int i = 0; i < nn->depth - 1; i++) {
        freeTensor(wb[0][i]);
        freeTensor(wb[1][i]);
    }
    free(wb[0]); free(wb[1]);
    free(wb);
}

void backProp(NeuralNet* nn, NNWeightsBiases* wb, Tensor* yTrue) {
    unsigned int layers = nn->depth;

    Tensor** wDeltas = wb[0];
    Tensor** bDeltas = wb[1];

    // delCost / delBias = (delCost / delA) * (delA / delZ) = simoid'(z) * 2(A_L - y)
    // delCost / delBias = (delCost / delA) * (delA / delZ) * (delZ / delW) = a_(L - 1) * simoid'(z) * 2(A_L - y)

    // delCost / delA
    // Derivative of loss function
    nn->lossFunction.lossDerivative(nn->layers[layers - 1], yTrue, bDeltas[layers - 2]);

    // delA / delZ
    Tensor* z = dupeTensor(nn->zs[layers - 1]);
    sigmoid_prime(z);
    mult(bDeltas[layers - 2], z, bDeltas[layers - 2]);

    // delZ / delW
    Tensor* a = dupeTensor(nn->layers[layers - 2]);
    Tensor* at = transpose(a);
    matmul(bDeltas[layers - 2], at, wDeltas[layers - 2]);
    freeTensor(a);
    freeTensor(at);

    for (int l = layers - 3; l >= 0; l--) {
        Tensor* z = dupeTensor(nn->zs[l + 1]);
        sigmoid_prime(z);

        Tensor* w = dupeTensor(nn->weights[l + 1]);
        Tensor* wt = transpose(w);
        Tensor* a = dupeTensor(nn->layers[l]);
        Tensor* at = transpose(a);

        matmul(wt, bDeltas[l + 1], bDeltas[l]);
        mult(bDeltas[l], z, bDeltas[l]);

        matmul(bDeltas[l], at, wDeltas[l]);

        freeTensor(z); freeTensor(w); freeTensor(wt); freeTensor(a); freeTensor(at);
    }
}

void applyBackProp(NeuralNet* nn, NNWeightsBiases* wb, float lr) {

    for (int i = 0; i < nn->depth - 1; i++) {
        scalarmult(wb[0][i], -lr); // Scale weight shifts by learning rate
        scalarmult(wb[1][i], -lr); // Scale bias shifts by learning rate

        add(nn->weights[i], wb[0][i], nn->weights[i]);
        add(nn->biases[i], wb[1][i], nn->biases[i]);
    }
}