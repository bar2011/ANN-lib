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

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // inputs = inputs to softmax
  // correct = vector of correct indices, one for each batch
  // returns loss average for each batch
  std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs,
          const std::shared_ptr<const Math::VectorBase<float>> &correct);

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // inputs = inputs to softmax
  // correct = one-hot encoded matrix
  // Note: optimally, use the other forward function. It's more optimized
  // returns loss average for each batch
  virtual std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs,
          const std::shared_ptr<const Math::MatrixBase<float>> &correct);

  std::unique_ptr<Math::Matrix<float>> predictSoftmax(
      const std::shared_ptr<const Math::MatrixBase<float>> &inputs) const;

  // perform backward pass based on the given inputs and correct values in
  // forward pass
  std::shared_ptr<const Math::Matrix<float>> backward();

  // Calculate average plain accuracy based on calculated
  float accuracy() const;

  // Calculate average loss accross batches
  virtual float mean() const;

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const {
    return m_dinputs;
  }

  std::shared_ptr<const Math::Matrix<float>> softmaxOutput() const {
    return m_softmaxOutput;
  }

  std::shared_ptr<const Math::Vector<float>> lossOutput() const {
    return m_lossOutput;
  }

private:
  // No ownership of m_input by the class. Just a constant view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::shared_ptr<const Math::VectorBase<float>> m_correct{};

  std::shared_ptr<Math::Matrix<float>> m_softmaxOutput{
      std::make_shared<Math::Matrix<float>>()};
  std::shared_ptr<Math::Vector<float>> m_lossOutput{
      std::make_shared<Math::Vector<float>>()};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Loss
} // namespace ANN
