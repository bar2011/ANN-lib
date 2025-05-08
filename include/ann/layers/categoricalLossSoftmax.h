#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vectorBase.h"

namespace Layer {
// Categorical Cross-Entropy loss class
class CategoricalLossSoftmax {
public:
  // initialize member variables
  CategoricalLossSoftmax(unsigned short neuronNum, unsigned int batchNum);

  // Copy constructor deleted
  CategoricalLossSoftmax(const CategoricalLossSoftmax &other) = delete;

  // Move constructor
  CategoricalLossSoftmax(CategoricalLossSoftmax &&other) noexcept;

  // Copy assignment deleted
  CategoricalLossSoftmax &
  operator=(const CategoricalLossSoftmax &other) = delete;

  // Move assignment
  CategoricalLossSoftmax &operator=(CategoricalLossSoftmax &&other) noexcept;

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // inputs = inputs to softmax
  // correct = vector of correct indicies, one for each batch
  // returns softmax output
  std::shared_ptr<const Math::Matrix<float>> forward(
      const std::shared_ptr<const Math::MatrixBase<float>> &inputs,
      const std::shared_ptr<const Math::VectorBase<unsigned short>> &correct);

  // perform backward pass based on the given inputs and correct values in
  // forward pass
  std::shared_ptr<const Math::Matrix<float>> backward();

  // Calculate average plain accuracy based on calculated
  float accuracy() const;

  // Calculate average loss accross batches
  float mean() const;

  std::shared_ptr<const Math::Matrix<float>> dinputs() const {
    return m_dinputs;
  }

  std::shared_ptr<const Math::Matrix<float>> softmaxOutput() const {
    return m_softmaxOutput;
  }

  std::shared_ptr<const Math::Vector<float>> lossOutput() const {
    return m_lossOutput;
  }

private:
  // No ownership of m_input by the class. Just a constant view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::shared_ptr<const Math::VectorBase<unsigned short>> m_correct{};

  std::shared_ptr<Math::Matrix<float>> m_softmaxOutput{};
  std::shared_ptr<Math::Vector<float>> m_lossOutput{};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{};
};
} // namespace Layer
