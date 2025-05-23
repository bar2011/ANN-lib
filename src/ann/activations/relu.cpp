#include "ann/activations/relu.h"

#include <algorithm>

namespace Activations {

ReLU::ReLU(size_t inputNum, size_t batchNum)
    : m_output{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)},
      m_dinputs{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)} {};

ReLU::ReLU(ReLU &&other) noexcept
    : m_input{std::move(other.m_input)}, m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

ReLU &ReLU::operator=(ReLU &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

std::shared_ptr<const Math::Matrix<float>>
ReLU::forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass

  if (inputs->rows() != m_output->rows())
    m_output =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());

  m_output->transform(
      *inputs, [](float *out, const float *in) { *out = std::max(0.0f, *in); },
      std::nullopt, 1);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>>
ReLU::backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) {
  m_dinputs->transform(
      *dvalues, *m_output,
      [](float *din, const float *dval, const float *out) {
        *din = *dval * ((*out > 0) ? 1 : 0);
      },
      std::nullopt, 2);

  return m_dinputs;
}
} // namespace Activations
