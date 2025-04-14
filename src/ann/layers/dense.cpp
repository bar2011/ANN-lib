#include "ann/layers/dense.h"

#include "math/linear.h"
#include "math/matrix.h"
#include "math/random.h"
#include "math/vector.h"

#include "ann/exception.h"

namespace Layer {
Dense::Dense(size_t inputNum, size_t neuronNum, size_t batchNum)
    : m_input{new Math::Matrix<double>{batchNum, inputNum}},
      m_weights{new Math::Matrix<double>{
          neuronNum, inputNum,
          []() -> double { return 0.01 * Math::Random::getNormal(); }}},
      m_biases{new Math::Vector<double>{neuronNum}},
      m_output{new Math::Matrix<double>{batchNum, neuronNum}} {};

Dense::Dense(Dense &&other)
    : m_input{other.m_input}, m_weights{other.m_weights},
      m_biases{other.m_biases}, m_output{other.m_output} {
  other.m_input = nullptr;
  other.m_weights = nullptr;
  other.m_biases = nullptr;
  other.m_output = nullptr;
}

Dense &Dense::operator=(Dense &&other) {
  if (&other != this) {
    // Free current pointers
    delete m_input;
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

Dense::~Dense() {
  delete m_input;
  delete m_weights;
  delete m_biases;
  delete m_output;
}

void Dense::forward(const Math::Matrix<double> &inputs) {
  if (!m_input || !m_output) // Check if pointers are valid
    throw ANN::Exception{"Dense::forward(const Math::Matrix<double>&)",
                         "Dense layer not properly initialized."};
  *m_input = inputs; // Store input for later use (e.g., backpropagation)
  *m_output = Math::dotTranspose(inputs, *m_weights) + *m_biases;
}
} // namespace Layer
