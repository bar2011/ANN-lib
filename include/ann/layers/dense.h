#pragma once

#include "math/matrix.h"
#include "math/vector.h"

namespace Layer {
// Basic Dense layer class.
// I - number of inputs to layer.
// N - number of neurons in layer.
// B - number of batches
template <size_t I, size_t N, size_t B> class Dense {
public:
  // 0-init inputs, biases, and outputs
  // random init weights, in a normal distribution with mean 0 and normal
  // diviasion 1
  Dense();

  // Copy constructor deleted
  Dense(const Dense<I, N, B> &other) = delete;

  // Move constructor
  Dense(Dense<I, N, B> &&other);

  // Copy assignment deleted
  Dense &operator=(const Dense<I, N, B> &other) = delete;

  // Move assignment
  Dense &operator=(Dense<I, N, B> &&other);

  // Destructor
  ~Dense();

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  void forward(const Math::Matrix<double> &inputs);

  const Math::Matrix<double> &output() const { return *m_output; }

private:
  Math::Matrix<double> *m_inputs{new Math::Matrix<double>{B, I}};
  Math::Matrix<double> *m_weights{new Math::Matrix<double>{N, I}};
  Math::Vector<double> *m_biases{new Math::Vector<double>{N}};
  Math::Matrix<double> *m_output{new Math::Matrix<double>{B, N}};
};
} // namespace Layer

#include "dense.tpp"
