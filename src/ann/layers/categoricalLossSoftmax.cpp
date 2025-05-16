#include "ann/layers/categoricalLossSoftmax.h"

#include "utils/parallel.h"

#include <cmath>

namespace Layer {

CategoricalLossSoftmax::CategoricalLossSoftmax(unsigned short neuronNum,
                                               unsigned int batchNum)
    : m_softmaxOutput{std::make_shared<Math::Matrix<float>>(batchNum,
                                                            neuronNum)},
      m_lossOutput{std::make_shared<Math::Vector<float>>(batchNum)},
      m_dinputs{std::make_shared<Math::Matrix<float>>(batchNum, neuronNum)} {}

CategoricalLossSoftmax::CategoricalLossSoftmax(
    CategoricalLossSoftmax &&other) noexcept
    : m_softmaxOutput{std::move(other.m_softmaxOutput)},
      m_lossOutput{std::move(other.m_lossOutput)}, m_input{other.m_input},
      m_correct{other.m_correct}, m_dinputs{std::move(other.m_dinputs)} {}

CategoricalLossSoftmax &
CategoricalLossSoftmax::operator=(CategoricalLossSoftmax &&other) noexcept {
  if (this != &other) {
    m_input = std::move(other.m_input);
    m_correct = std::move(other.m_correct);
    m_softmaxOutput = std::move(other.m_softmaxOutput);
    m_lossOutput = std::move(other.m_lossOutput);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

std::shared_ptr<const Math::Matrix<float>> CategoricalLossSoftmax::forward(
    const std::shared_ptr<const Math::MatrixBase<float>> &inputs,
    const std::shared_ptr<const Math::VectorBase<unsigned short>> &correct) {
  m_input = inputs;
  m_correct = correct;

  auto softmaxOutput{m_softmaxOutput};
  auto lossOutput{m_lossOutput};

  constexpr float epsilon{1e-7};

  Utils::Parallel::parallelFor(
      m_softmaxOutput->rows(),
      [inputs, correct, softmaxOutput, lossOutput, epsilon](size_t batch) {
        // max value in batch (for exponentiated values to not explode)
        float maxValue = (*inputs)[batch, 0];
        for (size_t i{1}; i < inputs->cols(); ++i)
          if ((*inputs)[batch, i] > maxValue)
            maxValue = (*inputs)[batch, i];

        // sum exponentiated inputs
        float normalBase{};
        for (size_t i{}; i < inputs->cols(); ++i) {
          (*softmaxOutput)[batch, i] =
              std::exp(static_cast<float>((*inputs)[batch, i] - maxValue));
          normalBase += (*softmaxOutput)[batch, i];
        }

        // Normalize values
        for (size_t i{}; i < inputs->cols(); ++i)
          (*softmaxOutput)[batch, i] /= normalBase;

        // Calculate batch loss
        float val{std::clamp(
            (*softmaxOutput)[batch, static_cast<size_t>((*correct)[batch])],
            epsilon, 1 - epsilon)};
        (*lossOutput)[batch] = -std::log(val);
      });

  return m_softmaxOutput;
}

std::shared_ptr<const Math::Matrix<float>> CategoricalLossSoftmax::backward() {
  for (size_t i{}; i < m_dinputs->rows(); ++i) {
    for (size_t j{}; j < m_dinputs->cols(); ++j)
      (*m_dinputs)[i, j] = (*m_softmaxOutput)[i, j];
    (*m_dinputs)[i, (*m_correct)[i]] -= 1;
  }
  for (size_t i{}; i < m_dinputs->rows(); ++i)
    for (size_t j{}; j < m_dinputs->cols(); ++j)
      (*m_dinputs)[i, j] /= m_dinputs->rows();

  return m_dinputs;
}

float CategoricalLossSoftmax::accuracy() const {
  // Get prediction for each row
  std::unique_ptr<Math::Vector<size_t>> prediction{
      m_softmaxOutput->argmaxRow()};

  float correctPredictions{};
  for (size_t i{}; i < prediction->size(); ++i)
    if ((*prediction)[i] == (*m_correct)[i])
      ++correctPredictions;

  return correctPredictions / prediction->size();
}

float CategoricalLossSoftmax::mean() const {
  float outputSum{};

  for (size_t i{}; i < m_lossOutput->size(); ++i)
    outputSum += (*m_lossOutput)[i];

  return outputSum / m_lossOutput->size();
}
} // namespace Layer
