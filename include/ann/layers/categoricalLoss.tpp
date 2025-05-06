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
  if (!m_dinputs)
    m_dinputs =
        std::make_shared<Math::Matrix<double>>(inputs->rows(), inputs->cols());

  for (size_t batch{}; batch < m_output->size(); ++batch) {
    double val{(*inputs)[batch, static_cast<size_t>((*correct)[batch])]};
    if (val >= 1 - 1e-7)
      val = 1 - 1e-7;
    else if (val <= 1e-7)
      val = 1e-7;
    (*m_output)[batch] = -std::log(val);
  }

  return m_output;
}

template <typename I, typename C>
std::shared_ptr<const Math::Matrix<double>> CategoricalLoss<I, C>::backward() {
  for (size_t i{}; i < m_input->rows(); ++i)
    for (size_t j{}; j < m_input->cols(); ++j) {
      if ((*m_correct)[i] != j)
        (*m_dinputs)[i, j] = 0;
      else
        // Set the corresponding value to the derivative of the loss function
        // Divided by number or rows (which is number of batches) for some sort
        // of normalization (useful while optimizing)
        (*m_dinputs)[i, j] = -1 / (*m_input)[i, j] / m_input->rows();
    }

  return m_dinputs;
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
