#pragma once

#include "ann/activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

#include <memory>

namespace Optimizers {
class Optimizer;
class SGD;
class Adagrad;
} // namespace Optimizers

namespace Layer {
enum class WeightInit {
  Xavier,
  He,
  Random,
};
// Basic Dense layer class.
// I = input data type
class Dense {
public:
  Dense() = delete;

  // 0-init inputs, biases, and outputs
  // Maximum allowed neuron number - 65,535
  // Maximum allowed batch number - 4,294,967,295
  // Uses activation of linear as default.
  Dense(unsigned int inputNum, unsigned short neuronNum, unsigned int batchNum,
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
  std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs);

  // perform backward pass with given dvalues
  // dvalues = matrix of how each input of each batch impacts the output of the
  // network
  // saves gradients as member variables
  std::shared_ptr<Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues);

  std::shared_ptr<const Math::Matrix<float>> output() const { return m_output; }

  std::shared_ptr<Math::Matrix<float>> dinputs() const { return m_dinputs; }

  friend class Optimizers::Optimizer;
  friend class Optimizers::SGD;
  friend class Optimizers::Adagrad;

private:
  // No ownership of m_input by the class. Only a view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::unique_ptr<Math::Matrix<float>> m_weights{
      std::make_unique<Math::Matrix<float>>()};
  std::unique_ptr<Math::Vector<float>> m_biases{
      std::make_unique<Math::Vector<float>>()};
  std::shared_ptr<Math::Matrix<float>> m_output{
      std::make_shared<Math::Matrix<float>>()};
  std::unique_ptr<ANN::Activation> m_activation{nullptr};

  std::unique_ptr<Math::Matrix<float>> m_dweights{
      std::make_unique<Math::Matrix<float>>()};
  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
  std::unique_ptr<Math::Vector<float>> m_dbiases{
      std::make_unique<Math::Vector<float>>()};

  std::unique_ptr<Math::Matrix<float>> m_weightUpdates{
      std::make_unique<Math::Matrix<float>>()};
  std::unique_ptr<Math::Vector<float>> m_biasUpdates{
      std::make_unique<Math::Vector<float>>()};
};
} // namespace Layer
