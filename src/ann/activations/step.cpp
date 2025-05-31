#include "ann/activations/step.h"

namespace ANN {
namespace Activation {

Step::Step(Step &&other) noexcept
    : m_input{std::move(other.m_input)}, m_output{std::move(other.m_output)},
      m_dinputs{std::move(other.m_dinputs)} {}

Step &Step::operator=(Step &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = std::move(other.m_input);
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

std::shared_ptr<const Math::Matrix<float>>
Step::forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) {
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
      *inputs, [](float *out, const float *in) { *out = ((*out > 0) ? 1 : 0); },
      std::nullopt, 1);

  return m_output;
}

std::unique_ptr<Math::Matrix<float>> Step::predict(
    const std::shared_ptr<const Math::MatrixBase<float>> &inputs) const {
  auto output{
      std::make_unique<Math::Matrix<float>>(inputs->rows(), inputs->cols())};
  output->transform(
      *inputs, [](float *out, const float *in) { *out = ((*out > 0) ? 1 : 0); },
      std::nullopt, 1);

  return output;
}

std::shared_ptr<const Math::Matrix<float>>
Step::backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues) {
  // Auto initialized to 0 - no calculation needed
  return m_dinputs;
}
} // namespace Activation
} // namespace ANN
