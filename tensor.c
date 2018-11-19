#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "tensor.h"
#include "functions.h"

Tensor* newTensor(unsigned int rank, const unsigned int shape[]) {
    Tensor* tensor = malloc(sizeof(Tensor));
    tensor->rank = rank;
    tensor->shape = calloc(rank, sizeof(unsigned int));

    size_t elements = 1;
    for (int i = 0; i < rank; i++) {
        tensor->shape[i] = shape[i];
        elements *= shape[i];
    }
    tensor->size = elements;
    tensor->data = calloc(elements, sizeof(float));

    return tensor;
}

void freeTensor(Tensor* tensor) {
    free(tensor->shape);
    free(tensor->data);
    free(tensor);
}

bool shapeMatches(const Tensor* a, const Tensor* b) {
    if (a->rank != b->rank) {
        return false;
    }

    for (int i = 0; i < a->rank; i++) {
        if(a->shape[i] != b->shape[i]) {
            return false;
        }
    }

    return true;
}


void copyTensor(Tensor* src, Tensor* dest) {
    if (shapeMatches(src, dest)) {

        for (size_t e = 0; e < src->size; e++) {
            dest->data[e] = src->data[e];
        }

    } else {
        fprintf(stderr, "Shape of tensors for copyTensor are incompatible\n");
        exit(100);
    }
}

Tensor* dupeTensor(Tensor* src) {
    Tensor* dest = newTensor(src->rank, src->shape);
    copyTensor(src, dest);

    return dest;
}

bool tensorEqual(Tensor* a, Tensor* b, float epsilon) {
    if (!shapeMatches(a, b)) return false;

    for (size_t i = 0; i < a->size; i++) {
        if (fabsf(a->data[i] - b->data[i]) > epsilon) return false;
    }

    return true;
}

void randomize(Tensor* tensor, float min, float max) {
    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = (float) (rand() / (RAND_MAX + 1.0)) * (max - min) + min;
    }
}

float* getElement(Tensor* tensor, const unsigned int* index) {
    unsigned int rank = tensor->rank;
    float* ptr = tensor->data;

    for (int i = 1; i <= rank; i++) {
        size_t shift = subtensorSize(tensor, rank - i);
        ptr += index[i - 1] * shift;
    }

    return ptr;
}

size_t subtensorSize(const Tensor* tensor, unsigned int dimensions) {
    if (dimensions == tensor->rank) return tensor->size;

    size_t size = 1;
    for (int i = 1; i <= dimensions; i++) {
        size *= tensor->shape[tensor->rank - i];
    }

    return size;
}

void fnApply(Tensor* tensor, float (*func)(float)) {
    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = (*func)(tensor->data[i]);
    }
}

void add(const Tensor* a, const Tensor* b, Tensor* c) {

    if (shapeMatches(a, b) && shapeMatches(b, c)) {

        for (size_t i = 0; i < a->size; i++) {
            c->data[i] = a->data[i] + b->data[i];
        }

    } else {
        fprintf(stderr, "Shape of tensors for add are incompatible\n");
        exit(100);
    }
}

void sub(const Tensor* a, const Tensor* b, Tensor* c) {

    if (shapeMatches(a, b) && shapeMatches(b, c)) {

        for (size_t i = 0; i < a->size; i++) {
            c->data[i] = a->data[i] - b->data[i];
        }

    } else {
        fprintf(stderr, "Shape of tensors for sub are incompatible\n");
        exit(100);
    }
}

void mult(const Tensor* a, const Tensor* b, Tensor* c) {

    if (shapeMatches(a, b) && shapeMatches(b, c)) {

        for (size_t i = 0; i < a->size; i++) {
            c->data[i] = a->data[i] * b->data[i];
        }

    } else {
        fprintf(stderr, "Shape of tensors for mult are incompatible\n");
        exit(100);
    }
}

void scalarmult(Tensor* a, float x) {
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] = x * a->data[i];
    }
}

void matmul(const Tensor* a, const Tensor* b, Tensor* c) {
    int rank = a->rank;

    bool shapesValid = rank >= 2 && rank == b->rank && rank == c->rank;
   
    // Assert dimensions 0 to n - 2 match
    for (int i = 0; i < rank - 2; i++) {
        shapesValid = shapesValid && a->shape[i] == b->shape[i] && b->shape[i] == c->shape[i];
    }

    // Assert shape for final two dims is correct
    shapesValid = shapesValid && a->shape[rank - 1] == b->shape[rank - 2];
    shapesValid = shapesValid && c->shape[rank - 2] == a->shape[rank - 2];
    shapesValid = shapesValid && c->shape[rank - 1] == b->shape[rank - 1];

    if (shapesValid) {

        // Calculate the total matrices in the tensor
        unsigned int totalMatrices = 1;
        for (int i = 0; i < rank - 2; i++) {
            totalMatrices *= a->shape[i];
        }

        size_t a_matrix_size = subtensorSize(a, 2);
        size_t b_matrix_size = subtensorSize(b, 2);
        size_t c_matrix_size = subtensorSize(c, 2);

        float* aPtr = a->data;
        float* bPtr = b->data;
        float* cPtr = c->data;

        unsigned int aRows = a->shape[rank - 2];
        unsigned int aCols = a->shape[rank - 1];
        unsigned int bCols = b->shape[rank - 1];

        for (int i = 0; i < totalMatrices; i++) {

            // Perform the actual matrix multiplication
            for (int aRow = 0; aRow < aRows; aRow++) {
                for (int bCol = 0; bCol < bCols; bCol++) {
                    for (int aCol = 0; aCol < aCols; aCol++) {
                        // C[aRow][bCol] += A[aRow][aCol] * B[aCol][bCol]
                        *(cPtr + (aRow * bCols) + bCol) +=
                                *(aPtr + (aRow * aCols) + aCol)
                                * *(bPtr + (aCol * bCols) + bCol);
                    }
                }
            }

            // Move pointers to next matrices
            aPtr += a_matrix_size; bPtr += b_matrix_size; cPtr += c_matrix_size;
        }


    } else {
        fprintf(stderr, "Shape of tensors for matmul are incompatible\n");
        exit(100);
    }
}

Tensor* transpose(Tensor* a) {
    unsigned int rank = a->rank;

    if (rank >= 2) {

        unsigned int newShape[rank];

        for (int i = 0; i < rank - 2; i++) {
            newShape[i] = a->shape[i];
        }
        newShape[rank - 2] = a->shape[rank - 1];
        newShape[rank - 1] = a->shape[rank - 2];

        Tensor* new = newTensor(rank, newShape);


        // Calculate the total matrices in the tensor
        unsigned int totalMatrices = 1;
        for (int i = 0; i < rank - 2; i++) {
            totalMatrices *= a->shape[i];
        }

        size_t matrix_size = subtensorSize(new, 2);

        float* aPtr = a->data;
        float* ptr = new->data;

        for (int m = 0; m < totalMatrices; m++) {

            for (int i = 0; i < a->shape[rank - 2];  i++) {
                for (int j = 0; j < a->shape[rank - 1]; j++) {
                    *(ptr + j + (i * a->shape[rank - 1])) = *(aPtr + i + (j * a->shape[rank - 2]));
                }
            }

            // Move pointers to next matrices
            ptr += matrix_size;
            aPtr += matrix_size;
        }

        return new;

    } else {
        fprintf(stderr, "Shape of tensor for transpose is incompatible\n");
        exit(100);
    }
}

size_t argmax(Tensor* a) {
    float max = a->data[0];
    size_t maxIndex = 0;
    for (size_t i = 1; i < a->size; i++) {
        if (a->data[i] > max) {
            max = a->data[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

void sigmoid(Tensor* a) {
    fnApply(a, fnSigmoid);
}

void sigmoid_prime(Tensor* a) {
    fnApply(a, fnSigmoidPrime);
}

void softmax(Tensor* a) {
    if (a->size == 1) {
        sigmoid(a);
        return;
    }

    float expSum = 0;
    for (size_t i = 0; i < a->size; i++) {
        expSum += exp(a->data[i]);
    }
    for (size_t i = 0; i < a->size; i++) {
        a->data[i] = (float) exp(a->data[i]) / expSum;
    }
}


void printTensor(Tensor* tensor) {
    size_t size = subtensorSize(tensor, tensor->rank);

    printf("[");
    int ele;
    for (ele = 0; ele < size - 1; ele++) {
        printf("%f, ", tensor->data[ele]);
    }
    printf("%f]\n", tensor->data[ele]);
}

void printShape(Tensor* tensor) {
    printf("(");
    for (unsigned int i = 0; i < tensor->rank - 1; i++) {
        printf("%d, ", tensor->shape[i]);
    }
    printf("%d)\n", tensor->shape[tensor->rank - 1]);
}