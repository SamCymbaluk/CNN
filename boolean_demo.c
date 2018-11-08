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

float cost(NeuralNet* nn, Tensor* ytrue) {
    float c = 0;
    for (int i = 0; i < nn->output->size; i++) {
        c += (ytrue->data[i] - nn->output->data[i]) * (ytrue->data[i] - nn->output->data[i]);
    }

    return c;
}

int main() {
    srand(time(NULL));

    // Initialize neural network
    unsigned int nnShape[] = {2, 10, 10, 2};
    NeuralNet* nn = newNeuralNet(4, nnShape);
    randInit(nn);
    nn->train = true;

    printf("Initial (random) Evaluations:\n");
    for (int i = 0; i < 5; i++) {
        printf("\n");

        Tensor** xy = genData();
        copyTensor(xy[0], nn->input);
        forwardPass(nn);

        printf("Input: ");
        printTensor(xy[0], false);
        printf("Output: ");
        printTensor(nn->output, false);
        printf("Correct Output: ");
        printTensor(xy[1], false);
    }

    int batches = 100000;
    int batchSize = 4;
    int plotInterval = 1000;
    
    printf("\n---------------------\nTraining for %d batches\n---------------------\n\n", batches);

    printf("Press any key to begin training\n");
    getchar();
     
  
    for (int i = 0; i < batches; i++) {
        Tensor** xy = genData();
        copyTensor(xy[0], nn->input);
        forwardPass(nn);

        if (i % plotInterval == 0) {
            printf("Batches complete: %d/%d\n", i, batches);
        }

        freeTensor(xy[0]); freeTensor(xy[1]); free(xy);

        // Train on batch
        Tensor** batch[batchSize];
        for (int b = 0; b < batchSize; b++) batch[b] = genData();

        batchTrain(nn, batch, batchSize, 0.1);


        for (int b = 0; b < batchSize; b++) {
            freeTensor(batch[b][0]); freeTensor(batch[b][1]); free(batch[b]);
        }

    }
    printf("\n---------------------\nTraining complete\n---------------------\n\n");

    printf("Example Evaluations:\n");
    for (int i = 0; i < 5; i++) {
        printf("\n");

        Tensor** xy = genData();
        copyTensor(xy[0], nn->input);

        printf("Input: ");
        printTensor(xy[0], false);
        forwardPass(nn);
        printf("Output: ");
        printTensor(nn->output, false);
        printf("Correct Output: ");
        printTensor(xy[1], false);
    }

    getchar();
}

