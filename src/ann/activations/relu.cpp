#include "ann/activations/relu.h"

#include <algorithm>

namespace ANN {
namespace Activation {

ReLU::ReLU(ReLU &&other) noexcept
    : m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

ReLU &ReLU::operator=(ReLU &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Matrix<float> &
ReLU::forward(const Math::MatrixBase<float> &inputs) {
  // If m_output's size doesn't match inputs' size, resize all matrices
  if (m_output.rows() != inputs.rows() || m_output.cols() != inputs.cols()) {
    m_output = Math::Matrix<float>{inputs.rows(), inputs.cols()};
    m_dinputs = Math::Matrix<float>{inputs.rows(), inputs.cols()};
  }

  m_output.transform(
      inputs, [](float *out, const float *in) { *out = std::max(0.0f, *in); },
      std::nullopt, 1);

  return m_output;
}

Math::Matrix<float> ReLU::predict(const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs.rows(), inputs.cols()};

  output.transform(
      inputs, [](float *out, const float *in) { *out = std::max(0.0f, *in); },
      std::nullopt, 1);

  return output;
}

const Math::Matrix<float> &
ReLU::backward(const Math::MatrixBase<float> &dvalues) {
  m_dinputs.transform(
      dvalues, m_output,
      [](float *din, const float *dval, const float *out) {
        *din = *dval * ((*out > 0) ? 1 : 0);
      },
      std::nullopt, 2);

  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
