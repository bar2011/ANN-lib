#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Loss {
// Categorical Cross-Entropy loss class
class Binary : public Loss {
public:
  Binary() = default;

  virtual ~Binary() = default;

  // Copy constructor deleted
  Binary(const Binary &other) = delete;

  // Move constructor
  Binary(Binary &&other) noexcept;

  // Copy assignment deleted
  Binary &operator=(const Binary &other) = delete;

  // Move assignment
  Binary &operator=(Binary &&other) noexcept;

  // Forward pass: stores and returns layer output (average batch loss)
  // prediction = batch output of ANN
  // correct = "wanted" values of prediction
  virtual const Math::Vector<float> &
  forward(const Math::MatrixBase<float> &prediction,
          const Math::MatrixBase<float> &correct);

  // Backward pass: stores and returns input gradients
  virtual const Math::Matrix<float> &backward();

  // Calculate average plain accuracy based on calculated
  float accuracy() const;

private:
  // No ownership of m_input and m_correct by the class. Just a constant view.
  Math::MatrixView<float> m_predictions{};
  Math::MatrixView<float> m_correct{};
};
} // namespace Loss
} // namespace ANN
