#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Loss {
// Mean Squared Error loss class
class MSE : public Loss {
public:
  MSE() = default;

  virtual ~MSE() = default;

  // Copy constructor deleted
  MSE(const MSE &other) = delete;

  // Move constructor
  MSE(MSE &&other) noexcept;

  // Copy assignment deleted
  MSE &operator=(const MSE &other) = delete;

  // Move assignment
  MSE &operator=(MSE &&other) noexcept;

  // Forward pass: stores and returns layer output (average batch loss)
  // prediction = batch output of ANN
  // correct = "wanted" values of prediction
  virtual const Math::Vector<float> &
  forward(const Math::MatrixBase<float> &prediction,
          const Math::MatrixBase<float> &correct);

  // Backward pass: stores and returns input gradients
  virtual const Math::Matrix<float> &backward();

private:
  // No ownership of m_input and m_correct by the class. Just a constant view.
  Math::MatrixView<float> m_predictions{};
  Math::MatrixView<float> m_correct{};
};
} // namespace Loss
} // namespace ANN
