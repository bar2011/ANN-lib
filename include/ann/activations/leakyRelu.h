#pragma once

#include "activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Activation {

// Leaky Rectified Linear Unit activation function
class LeakyReLU : public Activation {
public:
  LeakyReLU(float alpha);

  // Copy constructor deleted
  LeakyReLU(const LeakyReLU &other) = delete;

  // Move constructor
  LeakyReLU(LeakyReLU &&other) noexcept;

  // Copy assignment deleted
  LeakyReLU &operator=(const LeakyReLU &other) = delete;

  // Move assignment
  LeakyReLU &operator=(LeakyReLU &&other) noexcept;

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  virtual std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs);

  // Compute dinputs gradient
  virtual std::shared_ptr<const Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues);

  virtual std::shared_ptr<const Math::Matrix<float>> output() const {
    return m_output;
  }

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const {
    return m_dinputs;
  }

  virtual bool isTrainable() const { return false; }

  virtual std::string_view name() const { return "LeakyReLU"; }

private:
  // No ownership of m_input by the class. Only a view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::shared_ptr<Math::Matrix<float>> m_output{
      std::make_shared<Math::Matrix<float>>()};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};

  float m_alpha{};
};
} // namespace Activation
} // namespace ANN
