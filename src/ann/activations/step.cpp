#include "ann/activations/step.h"

namespace ANN {
namespace Activation {

Step::Step(Step &&other) noexcept
    : m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

Step &Step::operator=(Step &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Matrix<float> &
Step::forward(const Math::MatrixBase<float> &inputs) {
  m_output = predict(inputs);
  return m_output;
}

Math::Matrix<float> Step::predict(const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs.rows(), inputs.cols()};
  output.transform(
      inputs, [](float *out, const float *in) { *out = ((*in > 0) ? 1 : 0); },
      std::nullopt, 1);

  return output;
}

const Math::Matrix<float> &
Step::backward(const Math::MatrixBase<float> &dvalues) {
  if (dvalues.rows() != m_dinputs.rows() || dvalues.cols() != m_dinputs.cols())
    m_dinputs = Math::Matrix<float>(dvalues.rows(), dvalues.cols());

  // Auto initialized to 0 - no calculation needed
  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
