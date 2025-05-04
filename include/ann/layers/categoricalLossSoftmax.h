#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vectorBase.h"

namespace Layer {
// Categorical Cross-Entropy loss class
// I = input type. Must be castable to double.
// C = correct indicies ty[e. Must be castable to size_t.
template <typename I = double, typename C = double>
class CategoricalLossSoftmax {
public:
  // initialize member variables
  CategoricalLossSoftmax(size_t neuronNum, size_t batchNum);

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
  std::shared_ptr<const Math::Matrix<double>>
  forward(const std::shared_ptr<const Math::MatrixBase<I>> &inputs,
          const std::shared_ptr<const Math::VectorBase<C>> &correct);

  // perform backward pass based on the given inputs and correct values in
  // forward pass
  std::shared_ptr<const Math::Matrix<double>> backward();

  // Calculate average plain accuracy based on calculated
  double accuracy() const;

  // Calculate average loss accross batches
  double mean() const;

  std::shared_ptr<const Math::Matrix<double>> dinputs() const {
    return m_dinputs;
  }

  std::shared_ptr<const Math::Matrix<double>> softmaxOutput() const {
    return m_softmaxOutput;
  }

  std::shared_ptr<const Math::Vector<double>> lossOutput() const {
    return m_lossOutput;
  }

private:
  // No ownership of m_input by the class. Just a constant view.
  std::shared_ptr<const Math::MatrixBase<I>> m_input{};
  std::shared_ptr<const Math::VectorBase<C>> m_correct{};

  std::shared_ptr<Math::Matrix<double>> m_softmaxOutput{};
  std::shared_ptr<Math::Vector<double>> m_lossOutput{};

  std::shared_ptr<Math::Matrix<double>> m_dinputs{};
};
} // namespace Layer

#include "categoricalLossSoftmax.tpp"
