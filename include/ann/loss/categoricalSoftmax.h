#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vectorBase.h"

namespace ANN {
namespace Loss {
// Categorical Cross-Entropy loss class
class CategoricalSoftmax : public Loss {
public:
  CategoricalSoftmax() = default;

  // Copy constructor deleted
  CategoricalSoftmax(const CategoricalSoftmax &other) = delete;

  // Move constructor
  CategoricalSoftmax(CategoricalSoftmax &&other) noexcept;

  // Copy assignment deleted
  CategoricalSoftmax &operator=(const CategoricalSoftmax &other) = delete;

  // Move assignment
  CategoricalSoftmax &operator=(CategoricalSoftmax &&other) noexcept;

  // Forward pass: stores and returns layer output (average batch loss)
  // inputs = inputs to softmax layer
  // correct = indicies correct matrix
  const Math::Vector<float> &forward(const Math::MatrixBase<float> &inputs,
                                     const Math::VectorBase<float> &correct);

  // Forward pass: stores and returns layer output (average batch loss)
  // inputs = input to softmax layer
  // correct = one-hot encoded correct matrix
  // Note: other forward function is more optimized. Optimally, it's the one
  //       which should be used.
  virtual const Math::Vector<float> &
  forward(const Math::MatrixBase<float> &inputs,
          const Math::MatrixBase<float> &correct);

  // Backward pass: stores and returns input gradients
  virtual const Math::Matrix<float> &backward();

  // Forward pass without storing layer outputs
  // inputs = input to softmax layer
  Math::Matrix<float>
  predictSoftmax(const Math::MatrixBase<float> &inputs) const;

  // Calculate average plain accuracy based on calculated
  float accuracy() const;

  // Calculate average loss accross batches
  virtual float mean() const;

  const Math::Matrix<float> &softmaxOutput() const { return m_softmaxOutput; }

private:
  // No ownership of m_input by the class. Just a constant view.
  Math::MatrixView<float> m_input{};
  Math::VectorView<float> m_correct{};

  Math::Matrix<float> m_softmaxOutput{};

  Math::Matrix<float> m_dinputs{};
};
} // namespace Loss
} // namespace ANN
