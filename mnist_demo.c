#include <stdio.h>
#include <time.h>
#include "tensor.h"
#include "neuralnet.h"
#include "loss_functions.h"
#include "mnist_dataset.h"

int main() {
    srand(time(NULL));

    Dataset mnist = MNIST("DATASETS/mnist/train-images",
                          "DATASETS/mnist/train-labels",
                          "DATASETS/mnist/test-images",
                          "DATASETS/mnist/test-labels");

    for (size_t i = 0; i < 5; i++) printMnistDatum(mnist.getTrainElement(i));
}

