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

/**
 * Construct a new Tensor
 * @param rank The dimensionality of the tensor
 * @param shape An array of length 'rank' denoting the scalar size of each axis
 * @return Pointer to newly constructed Tensor
 */
Tensor* newTensor(unsigned int rank, const unsigned int shape[]);

/**
 * Preforms the nesseary free calls to deallocate the heap space used for a Tensor.
 * Corresponds with a call to newTensor
 * @param tensor
 */
void freeTensor(Tensor* tensor);

/**
 * Returns true if the shape of the provided Tensors are identical
 * @param a
 * @param b
 * @return
 */
bool shapeMatches(const Tensor* a, const Tensor* b);

/**
 * Copy the elements from the source Tensor to the destination Tensor.
 * Both Tensors must have the same shape
 * @param src Source Tensor
 * @param dest Destination Tensor
 */
void copyTensor(Tensor* src, Tensor* dest);

/**
 * Create a deep copy of a Tensor and return the address of the Ccopy
 * @param src The Tensor to copy
 * @return A pointer to the new copy
 */
Tensor* dupeTensor(Tensor* src);

/**
 * Returns true if the two Tensors are equal in shape and data
 * @param a
 * @param b
 * @param epsilon The minimum tolerable difference on a single element
 * @return
 */
bool tensorEqual(Tensor* a, Tensor* b, float epsilon);

/**
 * Populate all elements in a tensor with elements in the range [min, max)
 * @param tensor The Tensor to randomly populate
 * @param min The minimum value
 * @param max The maximum value
 */
void randomize(Tensor* tensor, float min, float max);

/**
 * Access a element pointer by exact index
 * @param tensor The Tensor to access an element of
 * @param index An array of length 'rank' specifying the index for each dimension
 * @return A pointer to the element that allows read and write
 */
float* getElement(Tensor* tensor, const unsigned int* index);

/**
 * Return size of a subtensor in floats
 * @param tensor
 * @param dimensions The number of dimensions in subtensor starting from the last dimension
 * @return The size in floats
 */
size_t subtensorSize(const Tensor* tensor, unsigned int dimensions);

/**
 * Element-wise application of a function of type float -> float to an Tensor
 * @param tensor The Tensor to apply on
 * @param func A pointer to a float -> float function
 */
void fnApply(Tensor* tensor, float (*func)(float));

/**
 * Preforms element-wise addition of a and b and stores the result in c
 * All Tensors must have matching shapes
 * c = a + b
 * @param a First addition operand
 * @param b Second addition operand
 * @param c Destination Tensor
 */
void add(const Tensor* a, const Tensor* b, Tensor* c);


/**
 * Preforms element-wise subtraction of b from a and stores the result in c
 * All Tensors must have matching shapes
 * c = a - b
 * @param a First subtraction operand
 * @param b Second subtraction operand
 * @param c Destination Tensor
 */
void sub(const Tensor* a, const Tensor* b, Tensor* c);

/**
 * Preforms element-wise multiplication of a and b and stores the result in c
 * All Tensors must have matching shapes
 * c = a * b
 * @param a First multiplication operand
 * @param b Second multiplication operand
 * @param c Destination Tensor
 */
void mult(const Tensor* a, const Tensor* b, Tensor* c);

/**
 * Performs element-wise multiplication of a Tensor by a constant float value
 * @param a Tensor
 * @param x float value
 */
void scalarmult(Tensor* a, float x);

/**
 * Preforms matrix multiplication between a and b and stores the result in c
 * The last 2 dimensions of the Tensors are treated as matrices.
 * Ex: (2, 3, 4, 5) * (2, 3, 5, 1) multiplies 6 4x5 matrices with 6 5x1 matrices)
 * The first n - 2 dimensions must be identical shapes and the last two dimensions must follow standard matrix
 * multiplication rules
 * @param a
 * @param b
 * @param c
 */
void matmul(const Tensor* a, const Tensor* b, Tensor* c);

/**
 * Return the flattened index of the Tensor element of greatest value
 * @param a Tensor
 * @return Flattened index
 */
size_t argmax(Tensor* a);

/**
 * Construct a new matrix consisting of the transposes of the the matrices in the given Tensor
 * As in matmul, the last two dimensions of the Tensor are treated as matrices
 * @param a
 * @return Pointer to a new Tensor consisting of the transposes of Tensor a
 */
Tensor* transpose(Tensor* a);

/**
 * Take element-wise sigmoid of a tensor
 * S(x) = 1 / (1 + e^-x)
 * @param a
 */
void sigmoid(Tensor* a);


/**
 * Take element-wise derivative of sigmoid prime of a tensor
 * S'(x) = S(x) * (1 - S(x));
 * @param a
 */
void sigmoid_prime(Tensor* a);

/**
 * Apply softmax to a tensor
 * @param a
 */
void softmax(Tensor* a);

/**
 * Print the elements of a Tensor as a flattened array
 * @param tensor The Tensor to print
 */
void printTensor(Tensor* tensor);

/**
 * Print the shape of a Tensor as a tuple of length 'rank'
 * @param tensor
 */
void printShape(Tensor* tensor);

#endif //PROJECT_TENSOR_H
