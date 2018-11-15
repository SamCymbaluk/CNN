#include <stdio.h>
#include "tensor.h"
#include "loss_functions.h"


float meanSquaredError(Tensor* ypred, Tensor* ytrue) {
    float loss = 0;
    if (shapeMatches(ypred, ytrue)) {
        for (size_t i = 0; i < ypred->size; i++) {
            loss += (ytrue->data[i] - ypred->data[i]) * (ytrue->data[i] - ypred->data[i]);
        }
    } else {
        printf("Mismatch Tensor shapes in meanSquaredError");
        exit(100);
    }

    return loss;
}

void meanSquaredErrorDerivative(Tensor* ypred, Tensor* ytrue, Tensor* dest) {
    sub(ypred, ytrue, dest);
}

bool meanSquaredErrorCorrect(Tensor* ypred, Tensor* ytrue) {
    return argmax(ypred) == argmax(ytrue);
}

LossFunction MeanSquaredError = {
        .loss = meanSquaredError,
        .lossDerivative = meanSquaredErrorDerivative,
        .correct = meanSquaredErrorCorrect
};

