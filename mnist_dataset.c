#include <stdbool.h>
#include <time.h>
#include "mnist_dataset.h"

#define IMAGE_SIZE 784

    size_t trainSize;
    size_t testSize;
    Datum* trainData;
    Datum* testData;

/**
 * Convert unsigned int from Big Endian to Little Endian
 * @param x
 * @param n
 * @return
 */
unsigned int convertToLE(unsigned int value)
{
    return (((value & 0x000000FF) << 24) |
            ((value & 0x0000FF00) <<  8) |
            ((value & 0x00FF0000) >>  8) |
            ((value & 0xFF000000) >> 24));
}

size_t loadImages(char* source, Tensor*** dest) {
    FILE* imageFile = fopen(source, "rb");

    unsigned int n;
    unsigned char image[IMAGE_SIZE];
    unsigned int tensorShape[] = {IMAGE_SIZE, 1};

    // Read image amount
    fseek(imageFile, 4, SEEK_SET);
    fread(&n, 4, 1, imageFile);
    n = convertToLE(n);
    *dest = calloc(n, sizeof(Tensor*));

    // Skip unneeded data
    fseek(imageFile, 8, SEEK_CUR);

    // Read in images
    for (int img = 0; img < n; img++) {
        fread(image, IMAGE_SIZE, 1, imageFile);

        Tensor* imgTensor = newTensor(2, tensorShape);
        for (int c = 0; c < IMAGE_SIZE; c++) {
            imgTensor->data[c] = (float) ((float) image[c] / 255.0);
        }

        (*dest)[img] = imgTensor;
    }

    fclose(imageFile);

    return n;
}

size_t loadLabels(char* source, Tensor*** dest) {
    FILE* labelFile = fopen(source, "rb");

    unsigned int n;
    unsigned int tensorShape[] = {10, 1};

    // Read label amount
    fseek(labelFile, 4, SEEK_SET);
    fread(&n, 4, 1, labelFile);
    n = convertToLE(n);
    *dest = calloc(n, sizeof(Tensor*));

    for (int l = 0; l < n; l++) {
        unsigned char label;
        fread(&label, 1, 1, labelFile);

        Tensor* labelTensor = newTensor(2, tensorShape);
        unsigned int index[] = {0, 0};
        index[0] = label;
        *getElement(labelTensor, index) = 1;

        (*dest)[l] = labelTensor;
    }

    fclose(labelFile);

    return n;
}

void zipData(Tensor** X, Tensor** Y, Datum* dest, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dest[i] = (Datum) {
            .x = X[i],
            .y = Y[i]
        };
    }
    free(X);
    free(Y);
}

Datum getTrainElement(size_t element) {
    return trainData[element];
}

Datum getTestElement(size_t element) {
    return testData[element];
}

void shuffle() {
    size_t i;
    for (i = trainSize - 1; i > 0; i--) {
        size_t j = (unsigned int) (rand()*(i+1));
        Datum t = trainData[j];
        trainData[j] = trainData[i];
        trainData[i] = t;
    }
}


Dataset MNIST(char* trainImages, char* trainLabels, char* testImages, char* testLabels) {
    Tensor** trainX;
    Tensor** trainY;
    Tensor** testX;
    Tensor** testY;

    trainSize = loadImages(trainImages, &trainX);
    testSize = loadImages(testImages, &testX);
    loadLabels(trainLabels, &trainY);
    loadLabels(testLabels, &testY);

    trainData = calloc(trainSize, sizeof(Datum));
    testData = calloc(testSize, sizeof(Datum));

    zipData(trainX, trainY, trainData, trainSize);
    zipData(testX, testY, testData, testSize);

    return (Dataset) {
        .trainElements = trainSize,
        .getTrainElement = getTrainElement,
        .testElements = testSize,
        .getTestElement = getTestElement,
        .shuffle = shuffle
    };
}

bool datumValid(Datum datum) {
    return datum.x->rank == 2 && datum.x->shape[0] == IMAGE_SIZE && datum.x->shape[1] == 1
           && datum.y->rank == 2 && datum.y->shape[0] == 10 && datum.y->shape[1] == 1;
}

char printSymbol(float x) {
    if (x <= 0.01) {
        return ' ';
    } else if (x < 0.3) {
        return '.';
    } else if (x < 0.4) {
        return ',';
    } else if (x < 0.5) {
        return '-';
    } else if (x < 0.6) {
        return '+';
    } else if (x < 0.7) {
        return '=';
    } else if (x < 0.8) {
        return 'x';
    } else if (x < 0.9) {
        return 'X';
    } else {
        return 'M';
    }
}

void printMnistDatum(Datum datum) {
    if (datumValid(datum)) {
        printf("O~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~O\n");
        for (int i = 0; i < 28; i++) {
            printf("| ");
            for (int j = 0; j < 28; j++) {
                printf("%c", printSymbol(datum.x->data[i*28 + j]));
            }
            printf(" |\n");
        }

        int label = -1;
        for (int i = 0; i < 10; i++) {
            if (datum.y->data[i] == 1) {
                label = i;
                break;
            }
        }

        printf("O~~~~~~~~~ Number: %d ~~~~~~~~~~O\n", label);
    } else {
        printf("[printDatum]: Invalid datum\n");
        exit(100);
    }
}