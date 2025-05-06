#pragma once

#include "categoricalLossSoftmax.h"

namespace Layer {
template <typename I, typename C>
CategoricalLossSoftmax<I, C>::CategoricalLossSoftmax(size_t neuronNum,
                                                     size_t batchNum)
    : m_softmaxOutput{std::make_shared<Math::Matrix<double>>(batchNum,
                                                             neuronNum)},
      m_lossOutput{std::make_shared<Math::Vector<double>>(batchNum)},
      m_dinputs{std::make_shared<Math::Matrix<double>>(batchNum, neuronNum)} {}

template <typename I, typename C>
CategoricalLossSoftmax<I, C>::CategoricalLossSoftmax(
    CategoricalLossSoftmax &&other) noexcept
    : m_softmaxOutput{std::move(other.m_softmaxOutput)},
      m_lossOutput{std::move(other.m_lossOutput)}, m_input{other.m_input},
      m_correct{other.m_correct}, m_dinputs{std::move(other.m_dinputs)} {}

template <typename I, typename C>
CategoricalLossSoftmax<I, C> &CategoricalLossSoftmax<I, C>::operator=(
    CategoricalLossSoftmax &&other) noexcept {
  if (this != &other) {
    m_input = std::move(other.m_input);
    m_correct = std::move(other.m_correct);
    m_softmaxOutput = std::move(other.m_softmaxOutput);
    m_lossOutput = std::move(other.m_lossOutput);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

template <typename I, typename C>
std::shared_ptr<const Math::Matrix<double>>
CategoricalLossSoftmax<I, C>::forward(
    const std::shared_ptr<const Math::MatrixBase<I>> &inputs,
    const std::shared_ptr<const Math::VectorBase<C>> &correct) {
  m_input = inputs;
  m_correct = correct;

  for (size_t batch{}; batch < m_softmaxOutput->rows(); ++batch) {
    // max value in batch (for exponentiated values to not explode)
    I maxValue = (*inputs)[batch, 0];
    for (size_t i{1}; i < inputs->cols(); ++i)
      if ((*inputs)[batch, i] > maxValue)
        maxValue = (*inputs)[batch, i];

    // sum of exponentiated inputs
    double normalBase{};

    for (size_t i{}; i < inputs->cols(); ++i) {
      (*m_softmaxOutput)[batch, i] =
          std::exp(static_cast<double>((*inputs)[batch, i] - maxValue));
      normalBase += (*m_softmaxOutput)[batch, i];
    }

    // Normalize values
    for (size_t i{}; i < inputs->cols(); ++i)
      (*m_softmaxOutput)[batch, i] /= normalBase;
  }

  constexpr double epsilon{1e-7};

  for (size_t batch{}; batch < m_lossOutput->size(); ++batch) {
    double val{std::clamp(
        (*m_softmaxOutput)[batch, static_cast<size_t>((*correct)[batch])],
        epsilon, 1 - epsilon)};
    (*m_lossOutput)[batch] = -std::log(val);
  }

  return m_softmaxOutput;
}

template <typename I, typename C>
std::shared_ptr<const Math::Matrix<double>>
CategoricalLossSoftmax<I, C>::backward() {
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

template <typename I, typename C>
double CategoricalLossSoftmax<I, C>::accuracy() const {
  // Get prediction for each row
  std::unique_ptr<Math::Vector<size_t>> prediction{
      m_softmaxOutput->argmaxRow()};

  double correctPredictions{};
  for (size_t i{}; i < prediction->size(); ++i)
    if ((*prediction)[i] == (*m_correct)[i])
      ++correctPredictions;

  return correctPredictions / prediction->size();
}

template <typename I, typename C>
double CategoricalLossSoftmax<I, C>::mean() const {
  double outputSum{};

  for (size_t i{}; i < m_lossOutput->size(); ++i)
    outputSum += (*m_lossOutput)[i];

  return outputSum / static_cast<double>(m_lossOutput->size());
}
} // namespace Layer
