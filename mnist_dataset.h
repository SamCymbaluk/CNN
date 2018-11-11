#ifndef PROJECT_MNIST_DATASET_H
#define PROJECT_MNIST_DATASET_H

#include <stdbool.h>
#include "dataset.h"

Dataset MNIST(char* trainImages, char* trainLabels, char* testImages, char* testLabels);

void printMnistDatum(Datum datum);

#endif //PROJECT_MNIST_DATASET_H
