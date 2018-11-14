#ifndef PROJECT_TRAINER_H
#define PROJECT_TRAINER_H

#include <stdbool.h>
#include "neuralnet.h"
#include "dataset.h"
#include "optimizer.h"

struct TrainingInfo {
    size_t epochIndex;
    float loss;
    float accuracy;
};

typedef struct TrainingInfo TrainingInfo;

void train(NeuralNet* nn, Optimizer optimizer, Dataset dataset,
           size_t epochs, unsigned int batchSize, float learnRate, void (*epochCallback)(TrainingInfo));

void printEpochCallback(TrainingInfo trainingInfo);

#endif //PROJECT_TRAINER_H
