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

  // Copy constructor deleted
  Binary(const Binary &other) = delete;

  // Move constructor
  Binary(Binary &&other) noexcept;

  // Copy assignment deleted
  Binary &operator=(const Binary &other) = delete;

  // Move assignment
  Binary &operator=(Binary &&other) noexcept;

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // prediction = output of ANN
  // correct = matrix filled with the correct values for each prediction, where:
  //    true - 1, false - 0
  // returns average loss in each batch
  std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &prediction,
          const std::shared_ptr<const Math::MatrixBase<float>> &correct);

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
  std::shared_ptr<const Math::MatrixBase<float>> m_correct{};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Loss
} // namespace ANN
