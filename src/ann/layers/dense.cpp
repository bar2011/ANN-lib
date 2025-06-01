#include "ann/layers/dense.h"

#include "ann/exception.h"

#include "math/dot.h"
#include "math/linear.h"
#include "math/matrix.h"
#include "math/random.h"
#include "utils/exceptions.h"

namespace ANN {
namespace Layers {
Dense::Dense(unsigned int inputNum, unsigned int neuronNum,
             WeightInit initMethod, float l1Weight, float l1Bias,
             float l2Weight, float l2Bias)
    : m_biases{neuronNum}, m_dweights{inputNum, neuronNum},
      m_dbiases{neuronNum}, m_weightCache{inputNum, neuronNum},
      m_weightMomentums{inputNum, neuronNum}, m_biasCache{neuronNum},
      m_biasMomentums{neuronNum}, m_l1Weight{l1Weight}, m_l1Bias{l1Bias},
      m_l2Weight{l2Weight}, m_l2Bias{l2Bias} {
  switch (initMethod) {
  case WeightInit::Xavier:
    m_weights = Math::Matrix<float>{
        inputNum, neuronNum, [inputNum, neuronNum]() -> float {
          return std::sqrt(2.0 / (inputNum + neuronNum)) *
                 Math::Random::getNormal();
        }};
    break;
  case WeightInit::He:
    m_weights = Math::Matrix<float>{
        inputNum, neuronNum, [inputNum, neuronNum]() -> float {
          return std::sqrt(2.0 / inputNum) * Math::Random::getNormal();
        }};
    break;
  case WeightInit::Random:
    m_weights = Math::Matrix<float>{inputNum, neuronNum, []() -> float {
                                      return 0.01 * Math::Random::getNormal();
                                    }};
    break;
  }
};

Dense::Dense(Dense &&other)
    : m_input{std::move(other.m_input)}, m_weights{std::move(other.m_weights)},
      m_biases{std::move(other.m_biases)}, m_output{std::move(other.m_output)},
      m_dweights{std::move(other.m_dweights)} {};

Dense &Dense::operator=(Dense &&other) {
  if (&other != this) {
    // Move other's pointers`
    m_input = std::move(other.m_input);
    m_weights = std::move(other.m_weights);
    m_biases = std::move(other.m_biases);
    m_output = std::move(other.m_output);
  }
  return *this;
}

const Math::Matrix<float> &
Dense::forward(const Math::MatrixBase<float> &inputs) {
  m_input = inputs.view(); // Store input for later use by backward pass
  m_output = predict(inputs);

  return m_output;
}

Math::Matrix<float>
Dense::predict(const Math::MatrixBase<float> &inputs) const {
  return Math::dot(inputs, m_weights, true, true) + m_biases;
}

const Math::Matrix<float> &
Dense::backward(const Math::MatrixBase<float> &dvalues) {
  // Regular backprop
  m_dweights = Math::dotTA<float>(m_input, dvalues, true, true);
  m_dinputs = Math::dotTB<float>(dvalues, m_weights, true);

  Utils::Parallel::dynamicParallelFor(
      dvalues.cols(), dvalues.rows(),
      [&dvalues, &dbiases = m_dbiases](size_t i) {
        for (size_t j{}; j < dvalues.cols(); ++j)
          dbiases[j] += dvalues[i, j];
      });

  // Regularization backprop

  // L1 backprop (absolute value derivative)
  if (m_l1Weight > 0)
    Utils::Parallel::dynamicParallelFor(
        m_weights.cols() * 4, m_weights.rows(),
        [&weights = m_weights, regularizer = m_l1Weight](size_t i) {
          for (size_t j{}; j < weights.cols(); ++j)
            weights[i, j] += regularizer * ((weights[i, j] >= 0) ? 1 : -1);
        });
  if (m_l1Bias > 0)
    Utils::Parallel::dynamicParallelFor(
        4, m_biases.size(),
        [&biases = m_biases, regularizer = m_l1Bias](size_t i) {
          biases[i] += regularizer * ((biases[i] >= 0) ? 1 : -1);
        });

  // L2 backprop (squared value derivative)
  if (m_l2Weight > 0)
    Utils::Parallel::dynamicParallelFor(
        m_weights.cols() * 5, m_weights.rows(),
        [&weights = m_weights, regularizer = m_l2Weight](size_t i) {
          for (size_t j{}; j < weights.cols(); ++j)
            weights[i, j] += regularizer * 2 * weights[i, j];
        });
  if (m_l2Bias > 0)
    Utils::Parallel::dynamicParallelFor(
        5, m_biases.size(),
        [&biases = m_biases, regularizer = m_l2Bias](size_t i) {
          biases[i] += regularizer * 2 * biases[i];
        });

  return m_dinputs;
}

void Dense::loadWeights(Math::Matrix<float> &weights) {
  if (weights.rows() != m_weights.rows() || weights.cols() != m_weights.cols())
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Given weights don't match saved weights' dimensions"};

  m_weights = std::move(weights);
}

void Dense::loadBiases(Math::Vector<float> &biases) {
  if (biases.size() != m_biases.size())
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Given biases don't match saved biases' size"};

  m_biases = std::move(biases);
}
}
} // namespace Layers
} // namespace ANN
