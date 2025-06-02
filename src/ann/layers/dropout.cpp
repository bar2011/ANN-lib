#include "ann/layers/dropout.h"

#include "math/random.h"
#include "utils/parallel.h"

namespace ANN {
namespace Layers {

Dropout::Dropout(float dropout) : m_dropout{dropout} {};

Dropout::Dropout(Dropout &&other) noexcept
    : m_output{std::move(other.m_output)}, m_mask{std::move(other.m_mask)},
      m_dinputs{std::move(other.m_dinputs)} {}

Dropout &Dropout::operator=(Dropout &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_mask = std::move(other.m_mask);
    m_output = std::move(other.m_output);
    m_dinputs = std::move(other.m_dinputs);
    m_dropout = other.m_dropout;
  }
  return *this;
}

const Math::Matrix<float> &
Dropout::forward(const Math::MatrixBase<float> &inputs) {
  // If mask's size doesn't match, resize (via recreation) all the matrices
  if (m_mask.rows() != inputs.rows() || m_mask.cols() != inputs.cols()) {
    m_mask = Math::Matrix<float>(inputs.rows(), inputs.cols());
    m_output = Math::Matrix<float>(inputs.rows(), inputs.cols());
    m_dinputs = Math::Matrix<float>(inputs.rows(), inputs.cols());
  }

  auto dropoutBatch{[&inputs, &output = m_output, &mask = m_mask,
                     dropout = m_dropout](size_t batch) {
    for (size_t i{}; i < inputs.cols(); ++i) {
      // Mask is normalized bernoulli output (to control mean output sum)
      mask[batch, i] = Math::Random::getBernoulli(1 - dropout) / (1 - dropout);
      output[batch, i] = inputs[batch, i] * mask[batch, i];
    }
  }};

  Utils::Parallel::dynamicParallelFor(inputs.cols() * 7, inputs.rows(),
                                      dropoutBatch);

  return m_output;
}

Math::Matrix<float>
Dropout::predict(const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs.rows(), inputs.cols()};

  auto dropoutBatch{
      [&inputs, &output, dropout = m_dropout](size_t batch) {
        for (size_t i{}; i < inputs.cols(); ++i) {
          // Mask is normalized bernoulli output (to control mean output sum)
          float mask{Math::Random::getBernoulli(1 - dropout) / (1 - dropout)};
          output[batch, i] = inputs[batch, i] * mask;
        }
      }};

  Utils::Parallel::dynamicParallelFor(inputs.cols() * 7, inputs.rows(),
                                      dropoutBatch);

  return output;
}

const Math::Matrix<float> &
Dropout::backward(const Math::MatrixBase<float> &dvalues) {
  m_dinputs.transform(dvalues, m_mask,
                      [](float *din, const float *dval, const float *mask) {
                        *din = *dval * *mask;
                      });

  return m_dinputs;
}
} // namespace Layers
} // namespace ANN
