#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

namespace ANN {
// Forward declarations
namespace Layers {
class Dense;
}

namespace Loss {
// Base loss class - not instantiatable
class Loss {
public:
  // Forward pass: stores and returns layer output (average batch loss)
  // prediction = batch output of ANN
  // correct = "wanted" values of prediction
  virtual const Math::Vector<float> &
  forward(const Math::MatrixBase<float> &prediction,
          const Math::MatrixBase<float> &correct) = 0;

  // Backward pass: stores and returns input gradients
  virtual const Math::Matrix<float> &backward() = 0;

  // Calculate average loss from calculated output derived from member function
  virtual float mean() const;

  // Calculate layer regularization loss based on its learned + hyper parameters
  float regularizationLoss(const Layers::Dense &layer) const;

  virtual const Math::Vector<float> &output() const { return m_output; }
  virtual const Math::Matrix<float> &dinputs() const { return m_dinputs; };

protected:
  Loss() = default;

  // Move constructor
  Loss(Loss &&other) : m_output{std::move(other.m_output)} {}

  Math::Vector<float> m_output{};
  Math::Matrix<float> m_dinputs{};
};
} // namespace Loss
} // namespace ANN
