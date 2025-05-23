#include "ann/activations/step.h"

namespace Activations {

Step::Step(size_t inputNum, size_t batchNum)
    : m_output{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)},
      m_dinputs{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)} {};

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

  if (inputs->rows() != m_output->rows())
    m_output =
        std::make_shared<Math::Matrix<float>>(inputs->rows(), inputs->cols());

  m_output->transform(
      *inputs, [](float *out, const float *in) { *out = ((*out > 0) ? 1 : 0); },
      std::nullopt, 1);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>>
Step::backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) {
  // Auto initialized to 0 - no calculation needed
  return m_dinputs;
}
} // namespace Activations
