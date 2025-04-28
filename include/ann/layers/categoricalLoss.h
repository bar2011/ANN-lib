#pragma once

#include "loss.h"

#include "math/matrixBase.h"
#include "math/vectorBase.h"

namespace Layer {
// Categorical Cross-Entropy loss class
// I = input type. Must be castable to double.
// C = correct indicies ty[e. Must be castable to size_t.
template <typename I = double, typename C = double>
class CategoricalLoss : public Loss {
public:
  // initialize member variables
  CategoricalLoss(size_t batchNum);

  // Copy constructor deleted
  CategoricalLoss(const CategoricalLoss &other) = delete;

  // Move constructor
  CategoricalLoss(CategoricalLoss &&other) noexcept;

  // Copy assignment deleted
  CategoricalLoss &operator=(const CategoricalLoss &other) = delete;

  // Move assignment
  CategoricalLoss &operator=(CategoricalLoss &&other) noexcept;

  // perform forward pass with give batch
  // saves inputs and outputs in member variables
  // inputs = output of ANN
  // correct = vector of correct indicies, one for each batch
  // returns average loss in each batch
  std::shared_ptr<const Math::Vector<double>>
  forward(const std::shared_ptr<const Math::MatrixBase<I>> &inputs,
          const std::shared_ptr<const Math::VectorBase<C>> &correct);

private:
  // No ownership of m_input by the class. Just a constant view.
  std::weak_ptr<const Math::MatrixBase<I>> m_input{};
  std::weak_ptr<const Math::VectorBase<C>> m_correct{};
};
} // namespace Layer

#include "categoricalLoss.tpp"
