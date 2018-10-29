#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "neuralnet.h"

void testMatMul() {
    unsigned int shape1[] = {1000, 50, 50};
    unsigned int shape2[] = {1000, 50, 50};
    unsigned int resShape[] = {1000, 50, 50};

    Tensor* a = newTensor(3, shape1);
    Tensor* b = newTensor(3, shape2);
    Tensor* res = newTensor(3, resShape);

    // Initialize A

    unsigned int index[3] = {};
    for (index[0] = 0; index[0] < 1000; index[0]++) {
        for (index[1] = 0; index[1] < 50; index[1]++) {
            for (index[2] = 0; index[2] < 50; index[2]++) {
                *(getElement(a, index)) = index[0] + 1;
            }
        }
    }

    // Initialize B
    index[0] = index[1] = index[2] = 0;
    for (index[0] = 0; index[0] < 1000; index[0]++) {
        for (index[1] = 0; index[1] < 50; index[1]++) {
            for (index[2] = 0; index[2] < 50; index[2]++) {
                *(getElement(b, index)) = index[1] + index[2];
            }
        }
    }
    //printf("Tensor A:\n");
    //printTensor(a, false);

    //printf("Tensor B:\n");
    //printTensor(b, false);

    clock_t begin = clock();

    matmul(a, b, res);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    //printf("Tensor RES:\n");
    //printTensor(res, false);

    printf("Time spent: %lfs\n", time_spent);

    freeTensor(a);
    freeTensor(b);

}

void testForwardPass() {
    unsigned int shape[] = {784, 16, 16, 10};

    NeuralNet* nn = newNeuralNet(4, shape);
    randInit(nn);
    randomize(nn->input, -1.0f, 1.0f);
    printf("Input: ");
    printTensor(nn->input, false);
    printf("Output: ");
    printTensor(nn->output, false);

    printf("\n-> Forward Pass ->\n\n");
    forwardPass(nn);

    printf("Input: ");
    printTensor(nn->input, false);
    printf("Output: ");
    printTensor(nn->output, false);

    printf("\n-----------\n\n");

    freeNeuralNet(nn);
}

void testBackProp() {
    unsigned int shape[] = {784, 16, 16, 10};

    NeuralNet* nn = newNeuralNet(4, shape);

    randInit(nn);
    nn->train = true;

    randomize(nn->input, 0, 1);
    forwardPass(nn);

    unsigned int yShape[] = {10, 1};
    unsigned int element[] = {4, 0};

    Tensor* ytrue = newTensor(2, yShape);
    *getElement(ytrue, element) = 1;
    printf("ytrue: ");
    printTensor(ytrue, false);

    Tensor*** wb = backProp(nn, ytrue);
    printf("Done backprop\n");

    printf("WB: %p\n", wb);
    printf("WB[0] %p\n", wb[0]);

    for (int i = 0; i < 3; i++) {
        // printf("Layer %d weights: ", i + 1);
        // printTensor(wb[0][0], false);
        printf("Layer %d biases: ", i + 1);
        printTensor(wb[1][i], false);
        printf("Shape: ");
        printShape(wb[1][i]);
    }

    freeNeuralNet(nn);
    freeTensor(ytrue);

    printf("End of testBackProp\n");
}

void testTranspose() {
    unsigned int shape[] = {3, 5};
    Tensor* o = newTensor(2, shape);

    for (int i = 1; i <= 15; i++) o->data[i - 1] = i;

    printf("Original: ");
    printTensor(o, false);

    Tensor* t = transpose(o);

    printf("Transpose: ");
    printTensor(t, false);

    freeTensor(o);
    freeTensor(t);
}

int main()
{
    // testMatMul();
    // testTranspose();
    // testForwardPass();
    testBackProp();
}


