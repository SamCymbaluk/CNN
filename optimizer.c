#include "optimizer.h"
#include <stdio.h>

void SGD_datumOptimize(NeuralNet* nn, Tensor*** wb, float lr) {
}

void SGD_batchOptimize(NeuralNet* nn, Tensor*** wb, float lr) {
    applyBackProp(nn, wb, lr);
}

void SGD_epochOptimize(NeuralNet* nn, Tensor*** wb, float lr) {

}

Optimizer SGD = {
    .datumOptimize = SGD_datumOptimize,
    .batchOptimize = SGD_batchOptimize,
    .epochOptimize = SGD_epochOptimize
};