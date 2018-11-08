#ifndef PROJECT_LOSS_FUNCTIONS_H
#define PROJECT_LOSS_FUNCTIONS_H

#include "tensor.h"

struct LossFunction {
    // Calculate loss based on a single network output Tensor and a single ground truth Tensor
    float (*loss)(Tensor*, Tensor*);
    // Element-wise loss derivative. Takes ypred, ytrue, dest
    void (*lossDerivative)(Tensor*, Tensor*, Tensor*);
};

typedef struct LossFunction LossFunction;

extern LossFunction* MeanSquaredError;

#endif //PROJECT_LOSS_FUNCTIONS_H
