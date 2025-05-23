#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"

#include <memory>

namespace Activation {

class Activation {
public:
  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  virtual std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) = 0;

  // Compute dinputs gradient
  virtual std::shared_ptr<const Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> output() const = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const = 0;
};

} // namespace Activations
