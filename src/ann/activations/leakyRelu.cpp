#include "ann/activations/leakyRelu.h"

namespace ANN {
namespace Activation {

LeakyReLU::LeakyReLU(float alpha) : m_alpha{alpha} {};

LeakyReLU::LeakyReLU(LeakyReLU &&other) noexcept
    : m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

LeakyReLU &LeakyReLU::operator=(LeakyReLU &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Matrix<float> &
LeakyReLU::forward(const Math::MatrixBase<float> &inputs) {
  // If m_output's size doesn't match inputs' size, resize all matrices
  if (m_output.rows() != inputs.rows() || m_output.cols() != inputs.cols()) {
    m_output = Math::Matrix<float>{inputs.rows(), inputs.cols()};
    m_dinputs = Math::Matrix<float>{inputs.rows(), inputs.cols()};
  }

  m_output.transform(
      inputs,
      [alpha = m_alpha](float *out, const float *in) {
        *out = (*in > 0) ? *in : (alpha * *in);
      },
      std::nullopt, 2);

  return m_output;
}

Math::Matrix<float>
LeakyReLU::predict(const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs.rows(), inputs.cols()};

  output.transform(
      inputs,
      [alpha = m_alpha](float *out, const float *in) {
        *out = (*in > 0) ? *in : (alpha * *in);
      },
      std::nullopt, 2);

  return output;
}

const Math::Matrix<float> &
LeakyReLU::backward(const Math::MatrixBase<float> &dvalues) {
  m_dinputs.transform(
      dvalues, m_output,
      [alpha = m_alpha](float *din, const float *dval, const float *out) {
        *din = *dval * ((*out > 0) ? 1 : alpha);
      },
      std::nullopt, 2);

  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
