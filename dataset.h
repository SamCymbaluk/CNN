#include <stdio.h>
#include "tensor.h"

#ifndef PROJECT_DATASET_H
#define PROJECT_DATASET_H


struct Datum {
    Tensor* x;
    Tensor* y;
};

typedef struct Datum Datum;

struct Dataset {
    size_t trainElements;
    Datum (*getTrainElement)(size_t);

    size_t testElements;
    Datum (*getTestElement)(size_t);

    void (*shuffle)();
};

typedef struct Dataset Dataset;

#endif //PROJECT_DATASET_H
