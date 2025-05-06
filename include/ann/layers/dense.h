#pragma once

#include "ann/activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

#include <memory>

namespace Layer {
enum class WeightInit {
  Xavier,
  He,
  Random,
};
// Basic Dense layer class.
// I = input data type
template <typename I = double> class Dense {
public:
  Dense() = delete;

  // 0-init inputs, biases, and outputs
  // random init weights, in a normal distribution with mean 0 and normal
  // diviasion 1.
  // Uses activation of linear as default.
  Dense(size_t inputNum, size_t neuronNum, size_t batchNum,
        ANN::Activation activation = {ANN::Activation::Linear},
        WeightInit initMethod = WeightInit::Random);

  // Copy constructor deleted
  Dense(const Dense &other) = delete;

  // Move constructor
  Dense(Dense &&other);

  // Copy assignment deleted
  Dense &operator=(const Dense &other) = delete;

  // Move assignment
  Dense &operator=(Dense &&other);

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  // Returns output
  std::shared_ptr<const Math::Matrix<double>>
  forward(const std::shared_ptr<const Math::MatrixBase<I>> &inputs);

  // perform backward pass with given dvalues
  // dvalues = matrix of how each input of each batch impacts the output of the
  // network
  // saves gradients as member variables
  std::shared_ptr<Math::Matrix<double>>
  backward(const std::shared_ptr<const Math::MatrixBase<double>> &dvalues);

  std::shared_ptr<const Math::Matrix<double>> output() const {
    return m_output;
  }

  std::shared_ptr<Math::Matrix<double>> dinputs() const { return m_dinputs; }

private:
  // No ownership of m_input by the class. Only a view.
  std::shared_ptr<const Math::MatrixBase<I>> m_input{};
  std::unique_ptr<Math::Matrix<double>> m_weights{
      std::make_unique<Math::Matrix<double>>()};
  std::unique_ptr<Math::Vector<double>> m_biases{
      std::make_unique<Math::Vector<double>>()};
  std::shared_ptr<Math::Matrix<double>> m_output{
      std::make_shared<Math::Matrix<double>>()};
  std::unique_ptr<ANN::Activation> m_activation{nullptr};

  std::unique_ptr<Math::Matrix<double>> m_dweights{
      std::make_unique<Math::Matrix<double>>()};
  std::shared_ptr<Math::Matrix<double>> m_dinputs{
      std::make_shared<Math::Matrix<double>>()};
  std::unique_ptr<Math::Vector<double>> m_dbiases{
      std::make_unique<Math::Vector<double>>()};
};
} // namespace Layer

#include "dense.tpp"
