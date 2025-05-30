#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

#include <memory>

namespace ANN {
// Forward declarations
namespace Layers {
class Dense;
}

namespace Loss {
// Base loss class - not instantiatable
class Loss {
public:
  // Calculate average loss from calculated output derived from member function
  virtual float mean() const;

  virtual std::shared_ptr<const Math::Vector<float>> output() const {
    return m_output;
  }

  float regularizationLoss(const Layers::Dense &layer) const;

  virtual std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &prediction,
          const std::shared_ptr<const Math::MatrixBase<float>> &correct) = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> backward() = 0;

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const = 0;

protected:
  Loss() = default;

  // Move constructor
  Loss(Loss &&other) : m_output{other.m_output} { other.m_output.reset(); }

  std::shared_ptr<Math::Vector<float>> m_output{
      std::make_shared<Math::Vector<float>>()};
};
} // namespace Loss
} // namespace ANN
