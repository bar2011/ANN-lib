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

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // predictions = output of ANN
  // correct = vector of correct indicies, one for each batch
  // returns average loss in each batch
  std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &predictions,
          const std::shared_ptr<const Math::VectorBase<float>> &correct);

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
  std::shared_ptr<const Math::VectorBase<float>> m_correct{};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Loss
} // namespace ANN
