#pragma once

#include "math/matrix.h"

#include <memory>
#include <string_view>

namespace ANN {
// Base layer class. Inherited by all layers and activations
class Layer {
public:
  enum class Type { Dense, Dropout, Step, ReLU, LeakyReLU, Sigmoid, Softmax };

  virtual ~Layer() = default;

  // Function which forwards the given inputs, and returns the outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  // Side effects: saves inputs and outputs for use with backpropagation
  virtual std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) = 0;

  // Function which forwards the given inputs, and returns the outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  // Doesn't save inputs and outputs for backprop
  virtual std::unique_ptr<Math::Matrix<float>> predict(
      const std::shared_ptr<const Math::MatrixBase<float>> &inputs) const = 0;

  // Function which runs backprop on the given dvalues, and returns the
  // gradients in terms of its inputs
  // dvalues dimensions - (batch_num, neuron_num)
  // outputs dimensions - (batch_num, input_num)
  virtual std::shared_ptr<const Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues) = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> output() const = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const = 0;

  // If outputs `true`, layer is passable to an optimizer
  // If outputs `false`, layer isn't passable to an optimizer
  virtual bool isTrainable() const = 0;

  // Returns name of the layer (e.g. "Dense")
  virtual std::string_view name() const = 0;

  // Returns layer type (e.g. Type::Dense)
  virtual Type type() const = 0;
};
} // namespace ANN
