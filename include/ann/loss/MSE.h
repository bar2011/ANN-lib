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

  // Copy constructor deleted
  MSE(const MSE &other) = delete;

  // Move constructor
  MSE(MSE &&other) noexcept;

  // Copy assignment deleted
  MSE &operator=(const MSE &other) = delete;

  // Move assignment
  MSE &operator=(MSE &&other) noexcept;

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // prediction = output of ANN
  // correct = matrix filled with the correct values for each prediction
  // returns average loss in each batch
  virtual std::shared_ptr<const Math::Vector<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &prediction,
          const std::shared_ptr<const Math::MatrixBase<float>> &correct);

  // perform backward pass based on the given inputs and correct values in
  // forward pass
  virtual std::shared_ptr<const Math::Matrix<float>> backward();

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const {
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
