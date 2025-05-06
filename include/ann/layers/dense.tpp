#pragma once

#include "dense.h"

#include "math/linear.h"
#include "math/matrix.h"
#include "math/random.h"

namespace Layer {
template <typename I>
Dense<I>::Dense(size_t inputNum, size_t neuronNum, size_t batchNum,
                ANN::Activation activation, WeightInit initMethod)
    : m_biases{std::make_unique<Math::Vector<double>>(neuronNum)},
      m_output{std::make_shared<Math::Matrix<double>>(batchNum, neuronNum)},
      m_activation{std::make_unique<ANN::Activation>(std::move(activation))},
      m_dweights{std::make_unique<Math::Matrix<double>>(inputNum, neuronNum)},
      m_dinputs{std::make_shared<Math::Matrix<double>>(batchNum, neuronNum)},
      m_dbiases{std::make_unique<Math::Vector<double>>(neuronNum)} {
  switch (initMethod) {
  case WeightInit::Xavier:
    m_weights = std::make_unique<Math::Matrix<double>>(
        inputNum, neuronNum, [inputNum, neuronNum]() -> double {
          return std::sqrt(2.0 / (inputNum + neuronNum)) *
                 Math::Random::getNormal();
        });
    break;
  case WeightInit::He:
    m_weights = std::make_unique<Math::Matrix<double>>(
        inputNum, neuronNum, [inputNum]() -> double {
          return std::sqrt(2.0 / inputNum) * Math::Random::getNormal();
        });
    break;
  case WeightInit::Random:
    m_weights = std::make_unique<Math::Matrix<double>>(
        inputNum, neuronNum,
        []() -> double { return 0.01 * Math::Random::getNormal(); });
    break;
  }
};

template <typename I>
Dense<I>::Dense(Dense &&other)
    : m_input{std::move(other.m_input)}, m_weights{std::move(other.m_weights)},
      m_biases{std::move(other.m_biases)}, m_output{std::move(other.m_output)},
      m_dweights{std::move(other.m_dweights)} {};

template <typename I> Dense<I> &Dense<I>::operator=(Dense &&other) {
  if (&other != this) {
    // Move other's pointers`
    m_input = std::move(other.m_input);
    m_weights = std::move(other.m_weights);
    m_biases = std::move(other.m_biases);
    m_output = std::move(other.m_output);
  }
  return *this;
}

template <typename I>
std::shared_ptr<const Math::Matrix<double>>
Dense<I>::forward(const std::shared_ptr<const Math::MatrixBase<I>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass
  m_output = std::make_shared<Math::Matrix<double>>(
      Math::dot<double, I, double>(*inputs, *m_weights) + *m_biases);
  auto activationForward{m_activation->getForward()};
  m_output->fill(
      [activationForward](double *val) { *val = activationForward(*val); });

  return m_output;
}

template <typename I>
std::shared_ptr<Math::Matrix<double>> Dense<I>::backward(
    const std::shared_ptr<const Math::MatrixBase<double>> &dvalues) {
  auto dactivation{std::make_unique<Math::Matrix<double>>(*m_output)};
  auto backwardActivation{m_activation->getBackward()};
  dactivation->transform(*dvalues,
                         [&backwardActivation](double *a, const double *b) {
                           *a = backwardActivation(*a, *b);
                         });
  *m_dweights =
      Math::dotTranspose<double, I, double>(*m_input, *dactivation, true);
  *m_dinputs =
      Math::dotTranspose<double, double, double>(*dactivation, *m_weights);

  // Sum dvalues row-wise and set it to dbiases
  for (size_t i{}; i < dactivation->rows(); ++i)
    for (size_t j{}; j < dactivation->cols(); ++j)
      (*m_dbiases)[j] += (*dactivation)[i, j];

  return m_dinputs;
}
} // namespace Layer
