#pragma once

#include "loss.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace Layer {
// Categorical Cross-Entropy loss class
class BinaryLoss : public Loss {
public:
  BinaryLoss() = default;

  // Copy constructor deleted
  BinaryLoss(const BinaryLoss &other) = delete;

  // Move constructor
  BinaryLoss(BinaryLoss &&other) noexcept;

  // Copy assignment deleted
  BinaryLoss &operator=(const BinaryLoss &other) = delete;

  // Move assignment
  BinaryLoss &operator=(BinaryLoss &&other) noexcept;

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // prediction = output of ANN
  // correct = matrix filled with the correct values for each prediction, where:
  //    true - 1, false - 0
  // returns average loss in each batch
  std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &prediction,
          const std::shared_ptr<const Math::MatrixBase<bool>> &correct);

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
  std::shared_ptr<const Math::MatrixBase<float>> m_predictions{};
  std::shared_ptr<const Math::MatrixBase<bool>> m_correct{};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Layer
