#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "neuralnet.h"

/**
 * Gen XOR data
 * @return [input, ytrue]
 */
Tensor** genData() {
    unsigned int shape[] = {2, 1};
    Tensor** xy = calloc(2, sizeof(Tensor*));
    xy[0] = newTensor(2, shape);
    xy[1] = newTensor(2, shape);

    // Setup input
    unsigned int index[] = {0, 0};
    int x = rand() % 2;
    int y = rand() % 2;
    *getElement(xy[0], index) = x;
    index[0] = 1;
    *getElement(xy[0], index) = y;

    // Output
    int ytrue = x != y; // XOR

    index[0] = ytrue;
    *getElement(xy[1], index) = 1;

    return xy;
}



int main() {
    srand(time(NULL));

    // Initialize neural network
    unsigned int nnShape[] = {2, 10, 10, 2};
    NeuralNet* nn = newNeuralNet(4, nnShape);
    randInit(nn);
    nn->train = true;

    for (int i = 0; i < 100000; i++) {
        Tensor** xy = genData();
        copyTensor(xy[0], nn->input);

        printf("Input: ");
        printTensor(xy[0], false);
        forwardPass(nn);
        printf("Output: ");
        printTensor(nn->output, false);
        printf("Correct Output: ");
        printTensor(xy[1], false);

        freeTensor(xy[0]); freeTensor(xy[1]); free(xy);

        // Train on batch
        Tensor** batch[16];
        for (int b = 0; b < 16; b++) batch[b] = genData();

        batchTrain(nn, batch, 16, 1);


        for (int b = 0; b < 16; b++) {
            freeTensor(batch[b][0]); freeTensor(batch[b][1]); free(batch[b]);
        }

    }


}

