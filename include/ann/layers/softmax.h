#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace Layer {
// Softmax activation as a layer
class Softmax {
public:
  // Initialize empty output matrix
  Softmax(size_t neuronNum, size_t batchNum);

  // Copy constructor deleted
  Softmax(const Softmax &other) = delete;

  // Move constructor
  Softmax(Softmax &&other) noexcept;

  // Copy assignment deleted
  Softmax &operator=(const Softmax &other) = delete;

  // Move assignment
  Softmax &operator=(Softmax &&other) noexcept;

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs);

  // Compute dinputs gradient
  std::shared_ptr<const Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues);

  std::shared_ptr<const Math::Matrix<float>> output() const { return m_output; }

  std::shared_ptr<const Math::Matrix<float>> dinputs() const {
    return m_dinputs;
  }

private:
  // No ownership of m_input by the class. Only a view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::shared_ptr<Math::Matrix<float>> m_output{
      std::make_shared<Math::Matrix<float>>()};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Layer

#include "softmax.tpp"
