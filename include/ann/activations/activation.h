#pragma once

#include "../layer.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Activation {

class Activation : public Layer {
public:
  virtual ~Activation() = default;

  // Forward pass: stores and returns layer outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  virtual const Math::Matrix<float> &
  forward(const Math::MatrixBase<float> &inputs) = 0;

  // Forward pass without storing layer outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  virtual Math::Matrix<float>
  predict(const Math::MatrixBase<float> &inputs) const = 0;

  // Backward pass: stores parameters gradients and returns input gradients
  // dvalues dimensions - (batch_num, neuron_num)
  // outputs dimensions - (batch_num, input_num)
  virtual const Math::Matrix<float> &
  backward(const Math::MatrixBase<float> &dvalues) = 0;

  virtual const Math::Matrix<float> &output() const = 0;

  virtual const Math::Matrix<float> &dinputs() const = 0;

  // Determines if layer is passable to an optimizer
  virtual bool isTrainable() const = 0;

  // Returns name of the layer (e.g. "Dense")
  virtual std::string_view name() const = 0;

  // Returns layer type (e.g. Type::Dense)
  virtual Type type() const = 0;
};

} // namespace Activation
} // namespace ANN
