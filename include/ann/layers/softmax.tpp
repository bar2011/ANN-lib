#pragma once

#include "softmax.h"

#include "math/linear.h"

namespace Layer {
template <typename I>
Softmax<I>::Softmax(size_t neuronNum, size_t batchNum)
    : m_output{std::make_shared<Math::Matrix<double>>(batchNum, neuronNum)},
      m_dinputs{std::make_shared<Math::Matrix<double>>(batchNum, neuronNum)} {};

template <typename I>
Softmax<I>::Softmax(Softmax &&other) noexcept
    : m_input{std::move(other.m_input)}, m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

template <typename I>
Softmax<I> &Softmax<I>::operator=(Softmax &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

template <typename I>
std::shared_ptr<const Math::Matrix<double>>
Softmax<I>::forward(const std::shared_ptr<const Math::MatrixBase<I>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass

  for (size_t batch{}; batch < m_output->rows(); ++batch) {
    // max value in batch (for exponentiated values to not explode)
    I maxValue = (*inputs)[batch, 0];
    for (size_t i{1}; i < inputs->cols(); ++i)
      if ((*inputs)[batch, i] > maxValue)
        maxValue = (*inputs)[batch, i];

    // sum of exponentiated inputs
    double normalBase{};

    for (size_t i{}; i < inputs->cols(); ++i) {
      (*m_output)[batch, i] =
          std::exp(static_cast<double>((*inputs)[batch, i] - maxValue));
      normalBase += (*m_output)[batch, i];
    }

    // Normalize values
    for (size_t i{}; i < inputs->cols(); ++i)
      (*m_output)[batch, i] /= normalBase;
  }

  return m_output;
}

template <typename I>
std::shared_ptr<const Math::Matrix<double>> Softmax<I>::backward(
    const std::shared_ptr<const Math::Matrix<double>> &dvalues) {
  for (size_t i{}; i < dvalues->rows(); ++i)
    for (size_t j{}; j < dvalues->cols(); ++j) {
      double sum{};
      for (size_t k{}; k < dvalues->cols(); ++k)
        sum += (*dvalues)[i, k] * (((j == k) ? 1.0 : 0.0) - (*m_output)[i, k]);
      (*m_dinputs)[i, j] = (*m_output)[i, j] * sum;
    }

  return m_dinputs;
}
} // namespace Layer
