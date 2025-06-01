#pragma once

#include "../layer.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Layers {
// Dropout activation as a layer
class Dropout : public Layer {
public:
  // dropout - rate of neurons to *disable*
  Dropout(float dropout);

  // Copy constructor deleted
  Dropout(const Dropout &other) = delete;

  // Move constructor
  Dropout(Dropout &&other) noexcept;

  // Copy assignment deleted
  Dropout &operator=(const Dropout &other) = delete;

  // Move assignment
  Dropout &operator=(Dropout &&other) noexcept;

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
  virtual std::string_view name() const { return "Dropout"; }
  virtual Layer::Type type() const { return Layer::Type::Dropout; }

private:
  Math::Matrix<float> m_output{};

  Math::Matrix<float> m_mask{};
  float m_dropout{};

  Math::Matrix<float> m_dinputs{};
};
} // namespace Layers
} // namespace ANN
