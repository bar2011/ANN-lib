#pragma once

#include "activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Activation {

// Rectified Linear Unit activation function
class ReLU : public Activation {
public:
  ReLU() = default;

  // Copy constructor deleted
  ReLU(const ReLU &other) = delete;

  // Move constructor
  ReLU(ReLU &&other) noexcept;

  // Copy assignment deleted
  ReLU &operator=(const ReLU &other) = delete;

  // Move assignment
  ReLU &operator=(ReLU &&other) noexcept;

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
} // namespace Activation
} // namespace ANN
