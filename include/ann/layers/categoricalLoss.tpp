#pragma once
#include <cassert>

#include "categoricalLoss.h"

#include <cmath>

namespace Layer {
template <typename I, typename C>
CategoricalLoss<I, C>::CategoricalLoss(size_t batchNum) : Loss(batchNum){};

template <typename I, typename C>
CategoricalLoss<I, C>::CategoricalLoss(CategoricalLoss<I, C> &&other) noexcept
    : Loss(std::move(other)) {
  m_input = other.m_input;

  other.m_input.reset();
}

template <typename I, typename C>
CategoricalLoss<I, C> &
CategoricalLoss<I, C>::operator=(CategoricalLoss<I, C> &&other) noexcept {
  if (&other != this) {
    // Move other's pointers
    m_input = other.m_input;
    m_output = other.m_output;

    // Reset other's pointers
    other.m_input.reset();
    other.m_output.reset();
  }
  return *this;
}

template <typename I, typename C>
std::shared_ptr<const Math::Vector<double>> CategoricalLoss<I, C>::forward(
    const std::shared_ptr<const Math::MatrixBase<I>> &inputs,
    const std::shared_ptr<const Math::VectorBase<C>> &correct) {
  // Store arguments for later use by backpropagation
  m_input = inputs;
  m_correct = correct;

  for (size_t batch{}; batch < m_output->size(); ++batch) {
    double val{(*inputs)[batch, (*correct)[batch]]};
    if (val >= 1 - 1e-7)
      val = 1 - 1e-7;
    else if (val <= 1e-7)
      val = 1e-7;
    (*m_output)[batch] = -1 * std::log(val);
  }

  return m_output;
}

template <typename I, typename C>
double CategoricalLoss<I, C>::accuracy() const {
  // Get prediction for each row
  std::unique_ptr<Math::Vector<size_t>> prediction{m_input->argmaxRow()};

  double correctPredictions{};
  for (size_t i{}; i < prediction->size(); ++i)
    if ((*prediction)[i] == (*m_correct)[i])
      ++correctPredictions;

  return correctPredictions / prediction->size();
}
} // namespace Layer
