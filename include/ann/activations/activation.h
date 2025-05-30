#pragma once

#include "../layer.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

#include <memory>

namespace ANN {
namespace Activation {

class Activation : public Layer {
public:
  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  virtual std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) = 0;

  virtual std::shared_ptr<Math::Matrix<float>> predict(
      const std::shared_ptr<const Math::MatrixBase<float>> &inputs) const = 0;

  // Compute dinputs gradient
  virtual std::shared_ptr<const Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues) = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> output() const = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const = 0;

  virtual bool isTrainable() const = 0;

  virtual std::string_view name() const = 0;
};

} // namespace Activation
} // namespace ANN
