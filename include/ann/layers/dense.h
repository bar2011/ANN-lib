#pragma once

#include "ann/activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

namespace Layer {
// Basic Dense layer class.
// I = input data type
template <typename I = double> class Dense {
public:
  Dense() = delete;

  // 0-init inputs, biases, and outputs
  // random init weights, in a normal distribution with mean 0 and normal
  // diviasion 1.
  // Uses activation of linear as default.
  Dense(size_t inputNum, size_t neuronNum, size_t batchNum,
        ANN::Activation activation = {ANN::Activation::Linear});

  // Copy constructor deleted
  Dense(const Dense &other) = delete;

  // Move constructor
  Dense(Dense &&other);

  // Copy assignment deleted
  Dense &operator=(const Dense &other) = delete;

  // Move assignment
  Dense &operator=(Dense &&other);

  // Destructor
  ~Dense();

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  void forward(const Math::MatrixBase<I> &inputs);

  const Math::Matrix<double> &output() const { return *m_output; }

private:
  // No ownership of m_input by the class. Only a view.
  const Math::MatrixBase<I> *m_input{nullptr};
  Math::Matrix<double> *m_weights{new Math::Matrix<double>{}};
  Math::Vector<double> *m_biases{new Math::Vector<double>{}};
  Math::Matrix<double> *m_output{new Math::Matrix<double>{}};
  ANN::Activation *m_activation{nullptr};
};
} // namespace Layer

#include "dense.tpp"
