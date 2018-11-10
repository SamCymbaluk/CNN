#include <stdbool.h>
#include "mnist_dataset.h"

    Tensor* trainX[];
    Tensor* trainY[];
    Tensor* testX[];
    Tensor* testY[];

void loadData(char* trainImages, char* trainLabels, char* testImages, char* testLabels) {
    FILE* trainImagesFile = fopen(trainImages, "rb");
    FILE* trainLabelsFile = fopen(trainLabels, "rb");
    FILE* testImagesFile = fopen(testImages, "rb");
    FILE* testLabelsFile = fopen(testLabels, "rb");


}


Dataset* MNIST(char* trainImages, char* trainLabels, char* testImages, char* testLabels) {
    loadData(trainImages, trainLabels, testImages, testLabels);
}