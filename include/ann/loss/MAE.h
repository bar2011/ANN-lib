#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Loss {
// Mean Absolute Error loss class
class MAE : public Loss {
public:
  MAE() = default;

  // Copy constructor deleted
  MAE(const MAE &other) = delete;

  // Move constructor
  MAE(MAE &&other) noexcept;

  // Copy assignment deleted
  MAE &operator=(const MAE &other) = delete;

  // Move assignment
  MAE &operator=(MAE &&other) noexcept;

  // Forward pass: stores and returns layer output (average batch loss)
  // prediction = batch output of ANN
  // correct = "wanted" values of prediction
  virtual const Math::Vector<float> &
  forward(const Math::MatrixBase<float> &prediction,
          const Math::MatrixBase<float> &correct);

  // Backward pass: stores and returns input gradients
  virtual const Math::Matrix<float> &backward();

private:
  Math::MatrixView<float> m_predictions{};
  Math::MatrixView<float> m_correct{};
};
} // namespace Loss
} // namespace ANN
