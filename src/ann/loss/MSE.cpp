#include "ann/loss/MSE.h"

namespace ANN {
namespace Loss {

MSE::MSE(MSE &&other) noexcept : Loss(std::move(other)) {
  m_predictions = std::move(other.m_predictions);
  m_correct = std::move(other.m_correct);
  m_dinputs = std::move(other.m_dinputs);
}

MSE &MSE::operator=(MSE &&other) noexcept {
  if (&other != this) {
    m_predictions = std::move(other.m_predictions);
    m_output = std::move(other.m_output);
    m_correct = std::move(other.m_correct);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Vector<float> &
MSE::forward(const Math::MatrixBase<float> &predictions,
             const Math::MatrixBase<float> &correct) {
  // Store arguments for later use by backpropagation
  m_predictions = predictions.view();
  m_correct = correct.view();

  // If m_dinput's size doesn't match inputs' size, resize all matrices
  if (m_dinputs.rows() != predictions.rows() ||
      m_dinputs.cols() != predictions.cols()) {
    m_output = Math::Vector<float>{predictions.rows()};
    m_dinputs = Math::Matrix<float>{predictions.rows(), predictions.cols()};
  }

  // An estimation of the cost of each iteration in terms of integer addition
  const size_t cost{4 * predictions.cols()};

  auto calculateBatch{
      [&predictions, &correct, &output = m_output](size_t batch) {
        float lossSum{};
        for (size_t i{}; i < predictions.cols(); ++i) {
          // Calculate difference
          float diff{correct[batch, i] - predictions[batch, i]};
          // Square it
          lossSum += diff * diff;
        }

        output[batch] = lossSum / static_cast<float>(predictions.cols());
      }};

  Utils::Parallel::dynamicParallelFor(cost, predictions.rows(), calculateBatch);

  return m_output;
}

const Math::Matrix<float> &MSE::backward() {
  const size_t cost{5 * m_predictions.cols()};

  // Add cols() to account for average, add rows() to normalize sum
  const size_t normalization{m_predictions.rows() * m_predictions.cols()};

  auto calculateBatch{[this, normalization](size_t batch) {
    for (size_t i{}; i < m_predictions.cols(); ++i) {
      // Calculate gradient according to derivative
      m_dinputs[batch, i] = 2 *
                            (m_predictions[batch, i] - m_correct[batch, i]) /
                            static_cast<float>(normalization);
    }
  }};

  Utils::Parallel::dynamicParallelFor(cost, m_predictions.rows(),
                                      calculateBatch);

  return m_dinputs;
}
} // namespace Loss
} // namespace ANN
