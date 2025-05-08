#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vectorBase.h"

namespace Layer {
// Categorical Cross-Entropy loss class
class CategoricalLoss : public Loss {
public:
  // initialize member variables
  CategoricalLoss(size_t batchNum);

  // Copy constructor deleted
  CategoricalLoss(const CategoricalLoss &other) = delete;

  // Move constructor
  CategoricalLoss(CategoricalLoss &&other) noexcept;

  // Copy assignment deleted
  CategoricalLoss &operator=(const CategoricalLoss &other) = delete;

  // Move assignment
  CategoricalLoss &operator=(CategoricalLoss &&other) noexcept;

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // inputs = output of ANN
  // correct = vector of correct indicies, one for each batch
  // returns average loss in each batch
  std::shared_ptr<const Math::Vector<float>> forward(
      const std::shared_ptr<const Math::MatrixBase<float>> &inputs,
      const std::shared_ptr<const Math::VectorBase<unsigned short>> &correct);

  // perform backward pass based on the given inputs and correct values in
  // forward pass
  std::shared_ptr<const Math::Matrix<float>> backward();

  // Calculate average plain accuracy based on calculated
  float accuracy() const;

  std::shared_ptr<const Math::Matrix<float>> dinputs() const {
    return m_dinputs;
  }

private:
  // No ownership of m_input and m_correct by the class. Just a constant view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::shared_ptr<const Math::VectorBase<unsigned short>> m_correct{};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{};
};
} // namespace Layer
