// DO NOT INCLUDE THIS FILE - include dense.h instead
#pragma once

#include "dense.h"

#include "math/linear.h"
#include "math/matrix.h"
#include "math/random.h"
#include "math/vector.h"

namespace Layer {
template <size_t I, size_t N, size_t B>
Dense<I, N, B>::Dense()
    : m_weights{new Math::Matrix<double>{N, I, []() -> double {
                                           return 0.01 *
                                                  Math::Random::getNormal();
                                         }}} {};
template <size_t I, size_t N, size_t B>
Dense<I, N, B>::Dense(Dense<I, N, B> &&other)
    : m_inputs{other.m_inputs}, m_weights{other.m_weights},
      m_biases{other.m_biases}, m_output{other.m_output} {
  other.m_inputs = nullptr;
  other.m_weights = nullptr;
  other.m_biases = nullptr;
  other.m_output = nullptr;
}

template <size_t I, size_t N, size_t B>
Dense<I, N, B> &Dense<I, N, B>::operator=(Dense &&other) {
  if (other != *this) {
    // Free current pointers
    delete m_inputs;
    delete m_weights;
    delete m_biases;
    delete m_output;

    // Move other's pointers`
    m_inputs = other.m_inputs;
    m_weights = other.m_weights;
    m_biases = other.m_biases;
    m_output = other.m_output;

    // "Remove" other's pointers
    other.m_inputs = nullptr;
    other.m_weights = nullptr;
    other.m_biases = nullptr;
    other.m_output = nullptr;
  }
  return *this;
}

template <size_t I, size_t N, size_t B> Dense<I, N, B>::~Dense() {
  delete m_inputs;
  delete m_weights;
  delete m_biases;
  delete m_output;
}

template <size_t I, size_t N, size_t B>
void Dense<I, N, B>::forward(const Math::Matrix<double> &inputs) {
  if (m_inputs && m_output) { // Check if pointers are valid
    *m_inputs = inputs; // Store input for later use (e.g., backpropagation)
    *m_output = Math::dotTranspose(inputs, *m_weights) + *m_biases;
  } else {
    throw std::runtime_error("Dense layer not properly initialized.");
  }
}
} // namespace Layer
