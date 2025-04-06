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

  // perform forward pass on given inpus
  // save inputs and outputs in member variables
  void forward(const Math::Vector<double> &inputs);

  const Math::Vector<double> &output() const { return m_output; }

private:
  Math::Vector<double> m_inputs{I};
  Math::Matrix<double> m_weights{N, I};
  Math::Vector<double> m_biases{N};
  Math::Vector<double> m_output{N};
};
} // namespace Layer

#include "dense.tpp"
