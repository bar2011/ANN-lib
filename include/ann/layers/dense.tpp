#pragma once

#include "dense.h"

#include "math/linear.h"
#include "math/random.h"

namespace Layer {
template <typename I>
Dense<I>::Dense(size_t inputNum, size_t neuronNum, size_t batchNum,
                ANN::Activation activation)
    : m_weights{new Math::Matrix<double>{
          neuronNum, inputNum,
          []() -> double { return 0.01 * Math::Random::getNormal(); }}},
      m_biases{new Math::Vector<double>{neuronNum}},
      m_output{new Math::Matrix<double>{batchNum, neuronNum}},
      m_activation{new ANN::Activation{std::move(activation)}} {};

template <typename I>
Dense<I>::Dense(Dense &&other)
    : m_input{std::move(other.m_input)}, m_weights{std::move(other.m_weights)},
      m_biases{std::move(other.m_biases)},
      m_output{std::move(other.m_output)} {};

template <typename I> Dense<I> &Dense<I>::operator=(Dense &&other) {
  if (&other != this) {
    // Move other's pointers`
    m_input.swap(other.m_input);
    m_weights = std::move(other.m_weights);
    m_biases = std::move(other.m_biases);
    m_output = std::move(other.m_output);

    other.m_input.reset();
  }
  return *this;
}

template <typename I>
std::shared_ptr<const Math::Matrix<double>>
Dense<I>::forward(const std::shared_ptr<const Math::MatrixBase<I>> &inputs) {
  m_input = inputs; // Store input for later use by backward pass
  m_output = std::shared_ptr<Math::Matrix<double>>{new Math::Matrix<double>(
      Math::dotTranspose<double>(*inputs, *m_weights) + *m_biases)};
  auto activationForward{m_activation->getForward()};
  m_output->fill(
      [activationForward](double *val) { *val = activationForward(*val); });

  return m_output;
}
} // namespace Layer
