#pragma once

#include "softmax.h"

#include "math/linear.h"

namespace Layer {
template <typename I>
Softmax<I>::Softmax(size_t neuronNum, size_t batchNum)
    : m_output{new Math::Matrix<double>{batchNum, neuronNum}},
      m_dinputs{new Math::Matrix<double>{0, neuronNum}} {};

template <typename I>
Softmax<I>::Softmax(Softmax &&other) noexcept
    : m_input{other.m_input}, m_output{other.m_output} {
  other.m_input.reset();
  other.m_output.reset();
}

template <typename I>
Softmax<I> &Softmax<I>::operator=(Softmax &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = other.m_input;
    m_output = other.m_output;

    // Reset other's pointers
    other.m_input.reset();
    other.m_output.reset();
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
  for (size_t i{}; i < dvalues->rows(); ++i) {
    std::unique_ptr<Math::Matrix<double>> jacobianMatrix{
        new Math::Matrix<double>{dvalues->cols(), dvalues->cols()}};

    for (size_t row{}; row < dvalues->cols(); ++row)
      for (size_t col{}; col < dvalues->cols(); ++col)
        (*jacobianMatrix)[row, col] =
            ((row == col) ? 1 : 0) - (*dvalues)[row, col];

    m_dinputs->insertRow(Math::dot(*jacobianMatrix, *(*dvalues)[i]));
  }

  return m_dinputs;
}
} // namespace Layer
