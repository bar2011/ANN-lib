#include "ann/activations/sigmoid.h"

#include <cmath>

namespace ANN {
namespace Activation {

Sigmoid::Sigmoid(Sigmoid &&other) noexcept
    : m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

Sigmoid &Sigmoid::operator=(Sigmoid &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Matrix<float> &
Sigmoid::forward(const Math::MatrixBase<float> &inputs) {
  // If m_output's size doesn't match inputs' size, resize all matrices
  if (m_output.rows() != inputs.rows() || m_output.cols() != inputs.cols()) {
    m_output = Math::Matrix<float>{inputs.rows(), inputs.cols()};
    m_dinputs = Math::Matrix<float>{inputs.rows(), inputs.cols()};
  }

  m_output.transform(
      inputs,
      [](float *out, const float *in) { *out = 1 / (1 + std::exp(-*in)); },
      std::nullopt, 55);

  return m_output;
}

Math::Matrix<float>
Sigmoid::predict(const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs.rows(), inputs.cols()};

  output.transform(
      inputs,
      [](float *out, const float *in) { *out = 1 / (1 + std::exp(-*in)); },
      std::nullopt, 55);

  return output;
}

const Math::Matrix<float> &
Sigmoid::backward(const Math::MatrixBase<float> &dvalues) {
  m_dinputs.transform(
      dvalues, m_output,
      [](float *din, const float *dval, const float *out) {
        *din = *dval * *out * (1 - *out);
      },
      std::nullopt, 4);

  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
