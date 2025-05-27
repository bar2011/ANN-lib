#include "ann/activations/leakyRelu.h"

namespace ANN {
namespace Activation {

LeakyReLU::LeakyReLU(float alpha) : m_alpha{alpha} {};

LeakyReLU::LeakyReLU(LeakyReLU &&other) noexcept
    : m_input{std::move(other.m_input)}, m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

LeakyReLU &LeakyReLU::operator=(LeakyReLU &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

std::shared_ptr<const Math::Matrix<float>> LeakyReLU::forward(
    const std::shared_ptr<const Math::MatrixBase<float>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass

  // If m_output's size doesn't match inputs' size, resize all matrices
  if (m_output->rows() != inputs->rows() ||
      m_output->cols() != inputs->cols()) {
    m_output =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());
    m_dinputs =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());
  }

  m_output->transform(
      *inputs,
      [alpha = m_alpha](float *out, const float *in) {
        *out = (*in > 0) ? *in : (alpha * *in);
      },
      std::nullopt, 2);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>>
LeakyReLU::backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) {
  m_dinputs->transform(
      *dvalues, *m_output,
      [alpha = m_alpha](float *din, const float *dval, const float *out) {
        *din = *dval * ((*out > 0) ? 1 : alpha);
      },
      std::nullopt, 2);

  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
