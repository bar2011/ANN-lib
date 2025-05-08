#include "ann/layers/categoricalLoss.h"

#include <cmath>

namespace Layer {

CategoricalLoss::CategoricalLoss(size_t batchNum) : Loss(batchNum) {};

CategoricalLoss::CategoricalLoss(CategoricalLoss &&other) noexcept
    : Loss(std::move(other)) {
  m_input = other.m_input;

  other.m_input.reset();
}

CategoricalLoss &CategoricalLoss::operator=(CategoricalLoss &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);
  }
  return *this;
}

std::shared_ptr<const Math::Vector<float>> CategoricalLoss::forward(
    const std::shared_ptr<const Math::MatrixBase<float>> &inputs,
    const std::shared_ptr<const Math::VectorBase<unsigned short>> &correct) {
  // Store arguments for later use by backpropagation
  m_input = inputs;
  m_correct = correct;
  if (!m_dinputs)
    m_dinputs =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());

  constexpr float epsilon{1e-7};

  for (size_t batch{}; batch < m_output->size(); ++batch) {
    float val{
        std::clamp((*inputs)[batch, static_cast<size_t>((*correct)[batch])],
                   epsilon, 1 - epsilon)};
    (*m_output)[batch] = -std::log(val);
  }

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>> CategoricalLoss::backward() {
  for (size_t i{}; i < m_input->rows(); ++i)
    for (size_t j{}; j < m_input->cols(); ++j) {
      if ((*m_correct)[i] != j)
        (*m_dinputs)[i, j] = 0;
      else
        // Set the corresponding value to the derivative of the loss function
        // Divided by number or rows (which is number of batches) for some sort
        // of normalization (useful while optimizing)
        (*m_dinputs)[i, j] = -1 / (*m_input)[i, j] / m_input->rows();
    }

  return m_dinputs;
}

float CategoricalLoss::accuracy() const {
  // Get prediction for each row
  std::unique_ptr<Math::Vector<size_t>> prediction{m_input->argmaxRow()};

  float correctPredictions{};
  for (size_t i{}; i < prediction->size(); ++i)
    if ((*prediction)[i] == (*m_correct)[i])
      ++correctPredictions;

  return correctPredictions / prediction->size();
}
} // namespace Layer
