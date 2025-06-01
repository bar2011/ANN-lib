#pragma once

#include "activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Activation {
// Softmax activation as a layer
class Softmax : public Activation {
public:
  Softmax() = default;

  // Copy constructor deleted
  Softmax(const Softmax &other) = delete;

  // Move constructor
  Softmax(Softmax &&other) noexcept;

  // Copy assignment deleted
  Softmax &operator=(const Softmax &other) = delete;

  // Move assignment
  Softmax &operator=(Softmax &&other) noexcept;

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

  virtual std::string_view name() const { return "Softmax"; }

  virtual Layer::Type type() const { return Layer::Type::Softmax; }

private:
  // No ownership of m_input by the class. Only a view.
  Math::Matrix<float> m_output{};

  Math::Matrix<float> m_dinputs{};
};
} // namespace Activation
} // namespace ANN
