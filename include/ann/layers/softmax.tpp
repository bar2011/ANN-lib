#pragma once

#include "softmax.h"

namespace Layer {
template <typename I>
Softmax<I>::Softmax(size_t neuronNum, size_t batchNum)
    : m_output{new Math::Matrix<double>{batchNum, neuronNum}} {};

template <typename I>
Softmax<I>::Softmax(Softmax &&other)
    : m_input{other.m_input}, m_output{other.m_output} {
  other.m_input = nullptr;
  other.m_output = nullptr;
}

template <typename I> Softmax<I> &Softmax<I>::operator=(Softmax &&other) {
  if (&other != this) {
    // Move other's pointers`
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);

    // "Remove" other's pointers
    other.m_input = nullptr;
    other.m_output = nullptr;
  }
  return *this;
}

template <typename I>
void Softmax<I>::forward(
    const std::shared_ptr<const Math::MatrixBase<I>> &inputs) {
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
}
} // namespace Layer
