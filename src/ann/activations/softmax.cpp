#include "ann/activations/softmax.h"

#include "utils/parallel.h"

#include <cmath>

namespace ANN {
namespace Activation {

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

  // If m_output's size doesn't match inputs' size, resize all matrices
  if (m_output->rows() != inputs->rows() ||
      m_output->cols() != inputs->cols()) {
    m_output =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());
    m_dinputs =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());
  }

  // An estimation of all the operations in a single iteration
  size_t cost{55 * inputs->cols()};

  auto calculateBatch{[inputs, output = m_output](size_t batch) {
    // max value in batch (for exponentiated values to not explode)
    float maxValue = (*inputs)[batch, 0];
    for (size_t i{1}; i < inputs->cols(); ++i)
      if ((*inputs)[batch, i] > maxValue)
        maxValue = (*inputs)[batch, i];

    // sum of exponentiated inputs
    float normalBase{};

    for (size_t i{}; i < inputs->cols(); ++i) {
      (*output)[batch, i] =
          std::exp(static_cast<float>((*inputs)[batch, i] - maxValue));
      normalBase += (*output)[batch, i];
    }

    // Normalize values
    for (size_t i{}; i < inputs->cols(); ++i)
      (*output)[batch, i] /= normalBase;
  }};

  Utils::Parallel::dynamicParallelFor(cost, m_output->rows(), calculateBatch);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>>
Softmax::backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) {
  // An estimation of all the operations in a single iteration
  size_t cost{dvalues->cols() * dvalues->cols() * 2};

  auto calculateBatch{
      [dvalues, dinputs = m_dinputs, output = m_output](size_t batch) {
        for (size_t j{}; j < dvalues->cols(); ++j) {
          float sum{};
          for (size_t k{}; k < dvalues->cols(); ++k)
            sum += (*dvalues)[batch, k] *
                   (((j == k) ? 1.0 : 0.0) - (*output)[batch, k]);
          (*dinputs)[batch, j] = (*output)[batch, j] * sum;
        }
      }};

  Utils::Parallel::dynamicParallelFor(cost, dvalues->rows(), calculateBatch);

  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
