#include "ann/loss/categorical.h"

#include <cmath>

namespace Loss {

Categorical::Categorical(Categorical &&other) noexcept
    : Loss(std::move(other)) {
  m_predictions = other.m_predictions;

  other.m_predictions.reset();
}

Categorical &Categorical::operator=(Categorical &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_predictions = std::move(other.m_predictions);
    m_output = std::move(other.m_output);
  }
  return *this;
}

std::shared_ptr<const Math::Vector<float>> Categorical::forward(
    const std::shared_ptr<const Math::MatrixBase<float>> &predictions,
    const std::shared_ptr<const Math::VectorBase<unsigned short>> &correct) {
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
  const size_t cost{50};

  constexpr float epsilon{1e-7};

  auto calculateBatch{
      [predictions, correct, output = m_output, epsilon](size_t batch) {
        float val{std::clamp(
            (*predictions)[batch, static_cast<size_t>((*correct)[batch])],
            epsilon, 1 - epsilon)};
        (*output)[batch] = -std::log(val);
      }};

  Utils::Parallel::dynamicParallelFor(cost, predictions->rows(),
                                      calculateBatch);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>> Categorical::backward() {
  for (size_t i{}; i < m_predictions->rows(); ++i)
    for (size_t j{}; j < m_predictions->cols(); ++j) {
      if ((*m_correct)[i] != j)
        (*m_dinputs)[i, j] = 0;
      else
        // Set the corresponding value to the derivative of the loss function
        // Divided by number or rows (which is number of batches) for some sort
        // of normalization (useful while optimizing)
        (*m_dinputs)[i, j] =
            -1 / (*m_predictions)[i, j] / m_predictions->rows();
    }

  return m_dinputs;
}

float Categorical::accuracy() const {
  // Get prediction for each row
  std::unique_ptr<Math::Vector<size_t>> prediction{m_predictions->argmaxRow()};

  float correctPredictions{};
  for (size_t i{}; i < prediction->size(); ++i)
    if ((*prediction)[i] == (*m_correct)[i])
      ++correctPredictions;

  return correctPredictions / prediction->size();
}
} // namespace Loss
