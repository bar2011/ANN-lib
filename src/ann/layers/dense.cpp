#include "ann/layers/dense.h"

#include "math/dot.h"
#include "math/linear.h"
#include "math/matrix.h"
#include "math/random.h"

namespace Layer {
Dense::Dense(unsigned int inputNum, unsigned short neuronNum,
             unsigned int batchNum, ANN::Activation activation,
             WeightInit initMethod)
    : m_biases{std::make_unique<Math::Vector<float>>(neuronNum)},
      m_output{std::make_shared<Math::Matrix<float>>(batchNum, neuronNum)},
      m_activation{std::make_unique<ANN::Activation>(std::move(activation))},
      m_dweights{std::make_unique<Math::Matrix<float>>(inputNum, neuronNum)},
      m_dinputs{std::make_shared<Math::Matrix<float>>(batchNum, neuronNum)},
      m_dbiases{std::make_unique<Math::Vector<float>>(neuronNum)},
      m_weightCache{std::make_unique<Math::Matrix<float>>(inputNum, neuronNum)},
      m_weightMomentums{
          std::make_unique<Math::Matrix<float>>(inputNum, neuronNum)},
      m_biasCache{std::make_unique<Math::Vector<float>>(neuronNum)},
      m_biasMomentums{std::make_unique<Math::Vector<float>>(neuronNum)} {
  switch (initMethod) {
  case WeightInit::Xavier:
    m_weights = std::make_unique<Math::Matrix<float>>(
        inputNum, neuronNum, [inputNum, neuronNum]() -> float {
          return std::sqrt(2.0 / (inputNum + neuronNum)) *
                 Math::Random::getNormal();
        });
    break;
  case WeightInit::He:
    m_weights = std::make_unique<Math::Matrix<float>>(
        inputNum, neuronNum, [inputNum]() -> float {
          return std::sqrt(2.0 / inputNum) * Math::Random::getNormal();
        });
    break;
  case WeightInit::Random:
    m_weights = std::make_unique<Math::Matrix<float>>(
        inputNum, neuronNum,
        []() -> float { return 0.01 * Math::Random::getNormal(); });
    break;
  }
};

Dense::Dense(Dense &&other)
    : m_input{std::move(other.m_input)}, m_weights{std::move(other.m_weights)},
      m_biases{std::move(other.m_biases)}, m_output{std::move(other.m_output)},
      m_dweights{std::move(other.m_dweights)} {};

Dense &Dense::operator=(Dense &&other) {
  if (&other != this) {
    // Move other's pointers`
    m_input = std::move(other.m_input);
    m_weights = std::move(other.m_weights);
    m_biases = std::move(other.m_biases);
    m_output = std::move(other.m_output);
  }
  return *this;
}

std::shared_ptr<const Math::Matrix<float>>
Dense::forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass
  m_output = std::make_shared<Math::Matrix<float>>(
      Math::dot(*inputs, *m_weights, true, true) + *m_biases);
  auto activationForward{m_activation->getForward()};
  m_output->fill(
      [activationForward](float *val) { *val = activationForward(*val); });

  return m_output;
}

std::shared_ptr<Math::Matrix<float>>
Dense::backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues) {
  auto dactivation{std::make_unique<Math::Matrix<float>>(*m_output)};
  auto backwardActivation{m_activation->getBackward()};
  dactivation->transform(*dvalues,
                         [&backwardActivation](float *a, const float *b) {
                           *a = backwardActivation(*a, *b);
                         });
  *m_dweights = Math::dotTA<float>(*m_input, *dactivation, true, true);
  *m_dinputs = Math::dotTB<float>(*dactivation, *m_weights, true);

  Utils::Parallel::dynamicParallelFor(
      dactivation->cols(), dactivation->rows(),
      [&dactivation, &dbiases = m_dbiases](size_t i) {
        for (size_t j{}; j < dactivation->cols(); ++j)
          (*dbiases)[j] += (*dactivation)[i, j];
      });

  return m_dinputs;
}
} // namespace Layer
