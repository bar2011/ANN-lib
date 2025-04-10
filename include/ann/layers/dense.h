#pragma once

#include "math/matrix.h"
#include "math/vector.h"

namespace Layer {
// Basic Dense layer class.
// I - number of inputs to layer.
// N - number of neurons in layer.
template <size_t I, size_t N> class Dense {
public:
  // 0-init inputs, biases, and outputs
  // random init weights, in a normal distribution with mean 0 and normal
  // diviasion 1
  Dense();

  // perform forward pass with given batch
  // save inputs and outputs in member variables
  void forward(const Math::Matrix<double> &inputs);

  const Math::Matrix<double> &output() const { return m_output; }

private:
  Math::Matrix<double> m_inputs{1, I};
  Math::Matrix<double> m_weights{N, I};
  Math::Vector<double> m_biases{N};
  Math::Matrix<double> m_output{N, 1};
};
} // namespace Layer

#include "dense.tpp"
