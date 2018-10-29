#ifndef PROJECT_TENSOR_H
#define PROJECT_TENSOR_H

#include <stdlib.h>
#include <stdbool.h>

struct Tensor {
    unsigned int rank;
    unsigned int* shape;
    size_t size;
    float* data;
};

typedef struct Tensor Tensor;

Tensor* newTensor(unsigned int rank, const unsigned int shape[]);

void freeTensor(Tensor* tensor);

void copyTensor(Tensor* src, Tensor* dest);

Tensor* dupeTensor(Tensor* src);

void randomize(Tensor* tensor, float min, float max);

float* getElement(Tensor* tensor, const unsigned int* index);

size_t subtensorSize(const Tensor* tensor, unsigned int dimensions);

void add(const Tensor* a, const Tensor* b, Tensor* c);

void sub(const Tensor* a, const Tensor* b, Tensor* c);

void mult(const Tensor* a, const Tensor* b, Tensor* c);

void scalarmult(Tensor* a, float x);

void matmul(const Tensor* a, const Tensor* b, Tensor* c);

Tensor* transpose(Tensor* a);

void sigmoid(Tensor* a);

void sigmoid_prime(Tensor* a);

void softmax(Tensor* a);

void printTensor(Tensor* tensor, bool addrs);

void printShape(Tensor* tensor);

#endif //PROJECT_TENSOR_H
