#include "ann/layers/softmax.h"

#include <cmath>

namespace Layer {

Softmax::Softmax(size_t neuronNum, size_t batchNum)
    : m_output{std::make_shared<Math::Matrix<float>>(batchNum, neuronNum)},
      m_dinputs{std::make_shared<Math::Matrix<float>>(batchNum, neuronNum)} {};

Softmax::Softmax(Softmax &&other) noexcept
    : m_input{std::move(other.m_input)}, m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

Softmax &Softmax::operator=(Softmax &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

std::shared_ptr<const Math::Matrix<float>>
Softmax::forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass

  for (size_t batch{}; batch < m_output->rows(); ++batch) {
    // max value in batch (for exponentiated values to not explode)
    float maxValue = (*inputs)[batch, 0];
    for (size_t i{1}; i < inputs->cols(); ++i)
      if ((*inputs)[batch, i] > maxValue)
        maxValue = (*inputs)[batch, i];

    // sum of exponentiated inputs
    float normalBase{};

    for (size_t i{}; i < inputs->cols(); ++i) {
      (*m_output)[batch, i] =
          std::exp(static_cast<float>((*inputs)[batch, i] - maxValue));
      normalBase += (*m_output)[batch, i];
    }

    // Normalize values
    for (size_t i{}; i < inputs->cols(); ++i)
      (*m_output)[batch, i] /= normalBase;
  }

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>>
Softmax::backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) {
  for (size_t i{}; i < dvalues->rows(); ++i)
    for (size_t j{}; j < dvalues->cols(); ++j) {
      float sum{};
      for (size_t k{}; k < dvalues->cols(); ++k)
        sum += (*dvalues)[i, k] * (((j == k) ? 1.0 : 0.0) - (*m_output)[i, k]);
      (*m_dinputs)[i, j] = (*m_output)[i, j] * sum;
    }

  return m_dinputs;
}
} // namespace Layer
