#include <math.h>

float fnSigmoid(float x) {
    return (float) (1.0 / (1.0 + exp(-x)));
}

float fnSigmoidPrime(float x) {
    return fnSigmoid(x) * (1 - fnSigmoid(x));
}