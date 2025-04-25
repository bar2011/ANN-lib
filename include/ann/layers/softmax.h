#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace Layer {
// Softmax activation as a layer
// I = input data type
template <typename I = double> class Softmax {
public:
  // Initialize empty output matrix
  Softmax(size_t neuronNum, size_t batchNum);

  // Copy constructor deleted
  Softmax(const Softmax &other) = delete;

  // Move constructor
  Softmax(Softmax &&other);

  // Copy assignment deleted
  Softmax &operator=(const Softmax &other) = delete;

  // Move assignment
  Softmax &operator=(Softmax &&other);

  // Destructor
  ~Softmax();

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  void forward(const Math::MatrixBase<I> &inputs);

  const Math::Matrix<double> &getOutput() const { return *m_output; }

private:
  // No ownership of m_input by the class. Only a view.
  const Math::MatrixBase<I> *m_input{nullptr};
  Math::Matrix<double> *m_output{new Math::Matrix<double>{}};
};
} // namespace Layer

#include "softmax.tpp"
