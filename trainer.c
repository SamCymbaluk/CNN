#include "trainer.h"

TrainingInfo postEpochBenchmarks(NeuralNet* nn, Dataset dataset, size_t epochIndex) {
    float loss = 0;
    size_t correctTotal = 0;

    for (size_t datumIndex = 0; datumIndex < dataset.testElements; datumIndex++) {
        Datum datum = dataset.getTestElement(datumIndex);

        // Move Datum x value to NN input and forward pass
        copyTensor(datum.x, nn->input);
        forwardPass(nn);

        bool correct = nn->lossFunction.correct(nn->output, datum.y);
        correct ? correctTotal++ : correctTotal;

        loss += nn->lossFunction.loss(nn->output, datum.y) / dataset.testElements;
    }

    float accuracy = (float) correctTotal / dataset.testElements;

    return (TrainingInfo) {
        .epochIndex = epochIndex,
        .loss = loss,
        .accuracy = accuracy
    };
}

void train(NeuralNet* nn, Optimizer optimizer, Dataset dataset,
        size_t epochs, unsigned int batchSize, float learnRate, void (*epochCallback)(TrainingInfo)) {

    NNWeightsBiases* datumWBUpdate = newWeightBiasUpdate(nn);
    NNWeightsBiases* batchWBUpdate = newWeightBiasUpdate(nn);
    NNWeightsBiases* epochWBUpdate = newWeightBiasUpdate(nn);

    Datum datum;

    for (size_t epochIndex = 0; epochIndex < epochs; epochIndex++) {

        /*
         * Train
         */
        for (size_t datumIndex = 0; datumIndex < dataset.trainElements; datumIndex++) {
            datum = dataset.getTrainElement(datumIndex);

            // Move Datum x value to NN input and forward pass
            copyTensor(datum.x, nn->input);
            forwardPass(nn);

            // Perform back propagation to get Weights Bias update for a single training example
            backProp(nn, datumWBUpdate, datum.y);

            // Optimize on single data point
            optimizer.datumOptimize(nn, datumWBUpdate, learnRate);

            // Aggregate WB updates for batch
            addWeightBiasUpdate(nn, batchWBUpdate, datumWBUpdate, batchWBUpdate);

            // Aggregate WB updates for epoch
            addWeightBiasUpdate(nn, epochWBUpdate, datumWBUpdate, epochWBUpdate);

            if ((datumIndex + 1) % batchSize == 0) {
                // Take average WB update over batch
                scaleWeightBiasUpdate(nn, batchWBUpdate, 1.0f / batchSize);

                // Optimize on batch
                optimizer.batchOptimize(nn, batchWBUpdate, learnRate);
                scaleWeightBiasUpdate(nn, batchWBUpdate, 0);
            }

            // Reset datumWBUpdate
            scaleWeightBiasUpdate(nn, datumWBUpdate, 0);
        }
        // Take average WB update over epoch
        scaleWeightBiasUpdate(nn, epochWBUpdate, 1.0f / dataset.trainElements);

        // Optimize on entire epoch
        optimizer.epochOptimize(nn, epochWBUpdate, learnRate);

        scaleWeightBiasUpdate(nn, batchWBUpdate, 0);
        scaleWeightBiasUpdate(nn, epochWBUpdate, 0);

        // Benchmarks
        TrainingInfo trainingInfo = postEpochBenchmarks(nn, dataset, epochIndex);
        epochCallback(trainingInfo);
    }

    freeWeightBiasUpdate(nn, datumWBUpdate);
    freeWeightBiasUpdate(nn, batchWBUpdate);
    freeWeightBiasUpdate(nn, epochWBUpdate);
}

void printEpochCallback(TrainingInfo trainingInfo) {
    printf("---------- Epoch: %zu ----------\n", trainingInfo.epochIndex);
    printf("Loss: %f\n", trainingInfo.loss);
    printf("Accuracy: %f\n\n", trainingInfo.accuracy);
}