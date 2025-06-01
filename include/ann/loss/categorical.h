#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vectorBase.h"

namespace ANN {
namespace Loss {
// Categorical Cross-Entropy loss class
class Categorical : public Loss {
public:
  Categorical() = default;

  // Copy constructor deleted
  Categorical(const Categorical &other) = delete;

  // Move constructor
  Categorical(Categorical &&other) noexcept;

  // Copy assignment deleted
  Categorical &operator=(const Categorical &other) = delete;

  // Move assignment
  Categorical &operator=(Categorical &&other) noexcept;

  // Forward pass: stores and returns layer output (average batch loss)
  // prediction = batch output of ANN
  // correct = indicies correct matrix
  const Math::Vector<float> &forward(const Math::MatrixBase<float> &predictions,
                                     const Math::VectorBase<float> &correct);

  // Forward pass: stores and returns layer output (average batch loss)
  // prediction = batch output of ANN
  // correct = one-hot encoded correct matrix
  // Note: other forward function is more optimized. Optimally, it's the one
  //       which should be used.
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
  Math::VectorView<float> m_correct{};
};
} // namespace Loss
} // namespace ANN
