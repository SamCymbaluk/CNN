#ifndef PROJECT_OPTIMIZER_H
#define PROJECT_OPTIMIZER_H

#include "tensor.h"
#include "neuralnet.h"

struct Optimizer {
    void (*datumOptimize)(NeuralNet* nn, Tensor *** wb, float lr);
    void (*batchOptimize)(NeuralNet* nn, Tensor *** wb, float lr);
    void (*epochOptimize)(NeuralNet* nn, Tensor *** wb, float lr);
};

typedef struct Optimizer Optimizer;

extern Optimizer SGD;

#endif //PROJECT_OPTIMIZER_H
