// DO NOT INCLUDE THIS FILE - include dense.h instead
#pragma once

#include "dense.h"

#include "math/linear.h"
#include "math/matrix.h"
#include "math/random.h"
#include "math/vector.h"

namespace Layer {
template <size_t I, size_t N>
Dense<I, N>::Dense()
    : m_weights(N, I,
                []() -> double { return 0.01 * Math::Random::getNormal(); }){};

template <size_t I, size_t N>
void Dense<I, N>::forward(const Math::Matrix<double> &inputs) {
  m_inputs = inputs; // Store input for later use (e.g., backpropagation)
  m_output = Math::dotTranspose(inputs, m_weights) +
             m_biases;
}
} // namespace Layer
