#include "ann/loss/categoricalSoftmax.h"

#include "utils/parallel.h"

#include <cmath>

namespace ANN {
namespace Loss {

CategoricalSoftmax::CategoricalSoftmax(CategoricalSoftmax &&other) noexcept
    : m_input{other.m_input}, m_correct{other.m_correct},
      m_softmaxOutput{std::move(other.m_softmaxOutput)},
      m_dinputs{std::move(other.m_dinputs)} {}

CategoricalSoftmax &
CategoricalSoftmax::operator=(CategoricalSoftmax &&other) noexcept {
  if (this != &other) {
    m_input = std::move(other.m_input);
    m_correct = std::move(other.m_correct);
    m_softmaxOutput = std::move(other.m_softmaxOutput);
    m_dinputs = std::move(other.m_dinputs);
  }
  return *this;
}

const Math::Vector<float> &
CategoricalSoftmax::forward(const Math::MatrixBase<float> &inputs,
                            const Math::VectorBase<float> &correct) {
  m_input = inputs.view();
  m_correct = correct.view();

  // If m_output's size doesn't match inputs' size, resize all matrices
  if (m_softmaxOutput.rows() != inputs.rows() ||
      m_softmaxOutput.cols() != inputs.cols()) {
    m_softmaxOutput = Math::Matrix<float>{inputs.rows(), inputs.cols()};
    m_output = Math::Vector<float>{inputs.rows()};
    m_dinputs = Math::Matrix<float>{inputs.rows(), inputs.cols()};
  }

  constexpr float epsilon{1e-7f};

  // Chose 70 as a rough summation of operations done in the loop in comparison
  // to a single addition
  const size_t cost{inputs.cols() * 70};

  Utils::Parallel::dynamicParallelFor(
      cost, m_softmaxOutput.rows(),
      [&inputs, &correct, &softmaxOutput = m_softmaxOutput, &output = m_output,
       epsilon](size_t batch) {
        // max value in batch (for exponentiated values to not explode)
        float maxValue = inputs[batch, 0];
        for (size_t i{1}; i < inputs.cols(); ++i)
          if (inputs[batch, i] > maxValue)
            maxValue = inputs[batch, i];

        // sum exponentiated inputs
        float normalBase{};
        for (size_t i{}; i < inputs.cols(); ++i) {
          softmaxOutput[batch, i] =
              std::exp(static_cast<float>(inputs[batch, i] - maxValue));
          normalBase += softmaxOutput[batch, i];
        }

        // Normalize values
        for (size_t i{}; i < inputs.cols(); ++i)
          softmaxOutput[batch, i] /= normalBase;

        // Calculate batch loss
        float val{std::clamp(
            softmaxOutput[batch, static_cast<size_t>(correct[batch])], epsilon,
            1 - epsilon)};
        output[batch] = -std::log(val);
      });

  return m_output;
}

const Math::Vector<float> &
CategoricalSoftmax::forward(const Math::MatrixBase<float> &inputs,
                            const Math::MatrixBase<float> &correct) {
  Math::Vector<float> correctVector{correct.rows()};

  for (size_t i{}; i < correct.rows(); ++i)
    for (size_t j{}; j < correct.cols(); ++j)
      if (correct[i, static_cast<size_t>(correctVector[i])] < correct[i, j])
        correctVector[i] = static_cast<float>(j);

  return forward(inputs, correctVector);
}

Math::Matrix<float> CategoricalSoftmax::predictSoftmax(
    const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs.rows(), inputs.cols()};

  // An estimation of all the operations in a single iteration
  size_t cost{55 * inputs.cols()};

  auto calculateBatch{[&inputs, &output](size_t batch) {
    // max value in batch (for exponentiated values to not explode)
    float maxValue = inputs[batch, 0];
    for (size_t i{1}; i < inputs.cols(); ++i)
      if (inputs[batch, i] > maxValue)
        maxValue = inputs[batch, i];

    // sum of exponentiated inputs
    float normalBase{};

    for (size_t i{}; i < inputs.cols(); ++i) {
      output[batch, i] =
          std::exp(static_cast<float>(inputs[batch, i] - maxValue));
      normalBase += output[batch, i];
    }

    // Normalize values
    for (size_t i{}; i < inputs.cols(); ++i)
      output[batch, i] /= normalBase;
  }};

  Utils::Parallel::dynamicParallelFor(cost, output.rows(), calculateBatch);

  return output;
}

const Math::Matrix<float> &CategoricalSoftmax::backward() {
  // Used as normalization base
  size_t batches{m_dinputs.rows()};

  // 3 = estimated cost of each single iteration in terms of integer addition
  size_t cost{m_dinputs.cols() * 3};

  // Implement derivative
  Utils::Parallel::dynamicParallelFor(
      cost, batches,
      [batches, &correct = m_correct, &dinputs = m_dinputs,
       &softmaxOutput = m_softmaxOutput](size_t i) {
        size_t correctIndex{static_cast<size_t>(correct[i])};
        for (size_t j{}; j < dinputs.cols(); ++j)
          dinputs[i, j] =
              (softmaxOutput[i, j] - ((j == correctIndex) ? 1 : 0)) /
              static_cast<float>(batches);
      });

  return m_dinputs;
}

float CategoricalSoftmax::accuracy() const {
  // Get prediction for each row
  auto prediction{m_softmaxOutput.argmaxRow()};

  float correctPredictions{};
  for (size_t i{}; i < prediction.size(); ++i)
    if (static_cast<float>(prediction[i]) == m_correct[i])
      ++correctPredictions;

  return correctPredictions / static_cast<float>(prediction.size());
}

float CategoricalSoftmax::mean() const {
  float outputSum{};

  for (size_t i{}; i < m_output.size(); ++i)
    outputSum += m_output[i];

  return outputSum / static_cast<float>(m_output.size());
}
} // namespace Loss
} // namespace ANN
