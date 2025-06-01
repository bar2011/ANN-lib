#pragma once

#include "activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Activation {

// Sigmoid activation function
class Sigmoid : public Activation {
public:
  Sigmoid() = default;

  // Copy constructor deleted
  Sigmoid(const Sigmoid &other) = delete;

  // Move constructor
  Sigmoid(Sigmoid &&other) noexcept;

  // Copy assignment deleted
  Sigmoid &operator=(const Sigmoid &other) = delete;

  // Move assignment
  Sigmoid &operator=(Sigmoid &&other) noexcept;

  // Forward pass: stores and returns layer outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  virtual const Math::Matrix<float> &
  forward(const Math::MatrixBase<float> &inputs);

  // Forward pass without storing layer outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  virtual Math::Matrix<float>
  predict(const Math::MatrixBase<float> &inputs) const;

  // Backward pass: stores parameters gradients and returns input gradients
  // dvalues dimensions - (batch_num, neuron_num)
  // outputs dimensions - (batch_num, input_num)
  virtual const Math::Matrix<float> &
  backward(const Math::MatrixBase<float> &dvalues);

  virtual const Math::Matrix<float> &output() const { return m_output; }
  virtual const Math::Matrix<float> &dinputs() const { return m_dinputs; }

  virtual bool isTrainable() const { return false; }
  virtual std::string_view name() const { return "Sigmoid"; }
  virtual Layer::Type type() const { return Layer::Type::Sigmoid; }

private:
  Math::Matrix<float> m_output{};

  Math::Matrix<float> m_dinputs{};
};
} // namespace Activation
} // namespace ANN
