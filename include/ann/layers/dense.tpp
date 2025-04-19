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
    : m_input{other.m_input}, m_weights{other.m_weights},
      m_biases{other.m_biases}, m_output{other.m_output} {
  other.m_input = nullptr;
  other.m_weights = nullptr;
  other.m_biases = nullptr;
  other.m_output = nullptr;
}

template <typename I> Dense<I> &Dense<I>::operator=(Dense &&other) {
  if (&other != this) {
    // Free current pointers
    delete m_weights;
    delete m_biases;
    delete m_output;

    // Move other's pointers`
    m_input = other.m_input;
    m_weights = other.m_weights;
    m_biases = other.m_biases;
    m_output = other.m_output;

    // "Remove" other's pointers
    other.m_input = nullptr;
    other.m_weights = nullptr;
    other.m_biases = nullptr;
    other.m_output = nullptr;
  }
  return *this;
}

template <typename I> Dense<I>::~Dense() {
  delete m_weights;
  delete m_biases;
  delete m_output;
}

template <typename I>
void Dense<I>::forward(const Math::MatrixBase<I> &inputs) {
  m_input = &inputs; // Store input for later use by backward pass
  *m_output = Math::dotTranspose<double>(inputs, *m_weights) + *m_biases;
  auto activationForward{m_activation->getForward()};
  m_output->fill(
      [activationForward](double *val) { *val = activationForward(*val); });
}
} // namespace Layer
