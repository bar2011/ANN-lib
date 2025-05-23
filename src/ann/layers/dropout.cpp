#include "ann/layers/dropout.h"

#include "math/random.h"
#include "utils/parallel.h"

namespace Layer {

Dropout::Dropout(unsigned int inputNum, unsigned int batchNum, float dropout)
    : m_output{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)},
      m_mask{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)},
      m_dropout{dropout},
      m_dinputs{std::make_shared<Math::Matrix<float>>(batchNum, inputNum)} {};

Dropout::Dropout(Dropout &&other) noexcept
    : m_mask{std::move(other.m_mask)}, m_output{std::move(other.m_output)},
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

std::shared_ptr<const Math::Matrix<float>>
Dropout::forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) {
  auto dropoutBatch{[&inputs, &output = m_output, &mask = m_mask,
                     dropout = m_dropout](size_t batch) {
    for (size_t i{}; i < inputs->cols(); ++i) {
      // Mask is normalized bernoulli output (to control mean output sum)
      (*mask)[batch, i] =
          Math::Random::getBernoulli(1 - dropout) / (1 - dropout);
      (*output)[batch, i] = (*inputs)[batch, i] * (*mask)[batch, i];
    }
  }};

  Utils::Parallel::dynamicParallelFor(inputs->cols() * 7, inputs->rows(),
                                      dropoutBatch);

  return m_output;
}

std::shared_ptr<const Math::Matrix<float>>
Dropout::backward(const std::shared_ptr<const Math::Matrix<float>> &dvalues) {
  m_dinputs->transform(*dvalues, *m_mask,
                       [](float *din, const float *dval, const float *mask) {
                         *din = *dval * *mask;
                       });

  return m_dinputs;
}
} // namespace Layer
