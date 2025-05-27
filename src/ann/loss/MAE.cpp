#include "ann/loss/MAE.h"

#include <cmath>

namespace ANN {
namespace Loss {

MAE::MAE(MAE &&other) noexcept : Loss(std::move(other)) {
  m_predictions = std::move(other.m_predictions);
  m_correct = std::move(other.m_correct);
  m_dinputs = std::move(other.m_dinputs);
}

MAE &MAE::operator=(MAE &&other) noexcept {
  if (&other != this) {
    m_predictions = std::move(other.m_predictions);
    m_output = std::move(other.m_output);
    m_correct = std::move(other.m_correct);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

std::shared_ptr<const Math::Vector<float>>
MAE::forward(const std::shared_ptr<const Math::MatrixBase<float>> &predictions,
             const std::shared_ptr<const Math::MatrixBase<float>> &correct) {
  // Store arguments for later use by backpropagation
  m_predictions = predictions;
  m_correct = correct;

  // If m_dinput's size doesn't match inputs' size, resize all matrices
  if (m_dinputs->rows() != predictions->rows() ||
      m_dinputs->cols() != predictions->cols()) {
    m_output = std::make_shared<Math::Vector<float>>(predictions->rows());
    m_dinputs = std::make_shared<Math::Matrix<float>>(predictions->rows(),
                                                      predictions->cols());
  }

  // An estimation of the cost of each iteration in terms of integer addition
  const size_t cost{4 * predictions->cols()};

  auto calculateBatch{[predictions, correct, output = m_output](size_t batch) {
    float lossSum{};
    for (size_t i{}; i < predictions->cols(); ++i) {
      // Calculate absolute difference
      lossSum += std::abs((*correct)[batch, i] - (*predictions)[batch, i]);
    }

    (*output)[batch] = lossSum / predictions->cols();
  }};

  Utils::Parallel::dynamicParallelFor(cost, predictions->rows(),
                                      calculateBatch);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>> MAE::backward() {
  const size_t cost{5 * m_predictions->cols()};

  // Add cols() to account for average, add rows() to normalize sum
  const size_t normalization{m_predictions->rows() * m_predictions->cols()};

  auto calculateBatch{[this, normalization](size_t batch) {
    for (size_t i{}; i < m_predictions->cols(); ++i) {
      float diff{(*m_predictions)[batch, i] - (*m_correct)[batch, i]};
      // Calculate gradient according to derivative
      (*m_dinputs)[batch, i] = ((diff >= 0) ? 1.0f : -1.0f) / normalization;
    }
  }};

  Utils::Parallel::dynamicParallelFor(cost, m_predictions->rows(),
                                      calculateBatch);

  return m_dinputs;
}
} // namespace Loss
} // namespace ANN
