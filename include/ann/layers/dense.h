#pragma once

#include "math/matrix.h"
#include "math/vector.h"

namespace Layer {
// Basic Dense layer class.
class Dense {
public:
  Dense() = delete;

  // 0-init inputs, biases, and outputs
  // random init weights, in a normal distribution with mean 0 and normal
  // diviasion 1
  Dense(size_t inputNum, size_t neuronNum, size_t batchNum);

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
  void forward(const Math::Matrix<double> &inputs);

  const Math::Matrix<double> &output() const { return *m_output; }

private:
  Math::Matrix<double> *m_input{new Math::Matrix<double>{}};
  Math::Matrix<double> *m_weights{new Math::Matrix<double>{}};
  Math::Vector<double> *m_biases{new Math::Vector<double>{}};
  Math::Matrix<double> *m_output{new Math::Matrix<double>{}};
};
} // namespace Layer
