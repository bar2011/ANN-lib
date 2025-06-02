#include "ann/loss/binary.h"

#include <cmath>

namespace ANN {
namespace Loss {

Binary::Binary(Binary &&other) noexcept : Loss(std::move(other)) {
  m_predictions = std::move(other.m_predictions);
  m_correct = std::move(other.m_correct);
  m_dinputs = std::move(other.m_dinputs);
}

Binary &Binary::operator=(Binary &&other) noexcept {
  if (&other != this) {
    m_predictions = std::move(other.m_predictions);
    m_output = std::move(other.m_output);
    m_correct = std::move(other.m_correct);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Vector<float> &
Binary::forward(const Math::MatrixBase<float> &predictions,
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
  const size_t cost{100 * predictions.cols()};

  constexpr float epsilon{1e-7f};

  auto calculateBatch{
      [&predictions, &correct, &output = m_output, epsilon](size_t batch) {
        float lossSum{};
        for (size_t i{}; i < predictions.cols(); ++i) {
          float predictionClamped{
              std::clamp(predictions[batch, i], epsilon, 1 - epsilon)};
          // Sum loss for both correct and incorrect options
          lossSum += -correct[batch, i] * std::log(predictionClamped) -
                     (1 - correct[batch, i]) * std::log(1 - predictionClamped);
        }
        output[batch] = lossSum / static_cast<float>(predictions.cols());
      }};

  Utils::Parallel::dynamicParallelFor(cost, predictions.rows(), calculateBatch);

  return m_output;
}

const Math::Matrix<float> &Binary::backward() {
  const size_t cost{5 * m_predictions.cols()};

  constexpr float epsilon{1e-7f};

  // Add cols() to account for average, add rows() to normalize sum
  const size_t normalization{m_predictions.rows() * m_predictions.cols()};

  auto calculateBatch{[this, normalization, epsilon](size_t batch) {
    for (size_t i{}; i < m_predictions.cols(); ++i) {
      float predictionClamped{
          std::clamp(m_predictions[batch, i], epsilon, 1 - epsilon)};
      // Calculate gradient according to derivative
      m_dinputs[batch, i] =
          -(m_correct[batch, i] / predictionClamped -
            (1 - m_correct[batch, i]) / (1 - predictionClamped)) /
          static_cast<float>(normalization);
    }
  }};

  Utils::Parallel::dynamicParallelFor(cost, m_predictions.rows(),
                                      calculateBatch);

  return m_dinputs;
}

float Binary::accuracy() const {
  float correctPredictions{};
  for (size_t batch{}; batch < m_predictions.rows(); ++batch)
    for (size_t i{}; i < m_predictions.cols(); ++i) {
      bool prediction{m_predictions[batch, i] >= 0.5};
      if (prediction == m_correct[batch, i])
        ++correctPredictions;
    }
  return correctPredictions /
         static_cast<float>(m_predictions.rows() * m_predictions.cols());
}
} // namespace Loss
} // namespace ANN
