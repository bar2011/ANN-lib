#pragma once

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

#include <memory>

// Forward declarations

namespace Optimizers {
class Optimizer;
class SGD;
class Adagrad;
class RMSProp;
class Adam;
} // namespace Optimizers

namespace Layer {
class Loss;
}

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
  Dense(unsigned int inputNum, unsigned short neuronNum,
        WeightInit initMethod = WeightInit::Random, float l1Weight = 0,
        float l1Bias = 0, float l2Weight = 0, float l2Bias = 0);

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
  friend class Optimizers::RMSProp;
  friend class Optimizers::Adam;

  friend class Loss;

private:
  // No ownership of m_input by the class. Only a view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::unique_ptr<Math::Matrix<float>> m_weights{
      std::make_unique<Math::Matrix<float>>()};
  std::unique_ptr<Math::Vector<float>> m_biases{
      std::make_unique<Math::Vector<float>>()};
  std::shared_ptr<Math::Matrix<float>> m_output{
      std::make_shared<Math::Matrix<float>>()};

  std::unique_ptr<Math::Matrix<float>> m_dweights{
      std::make_unique<Math::Matrix<float>>()};
  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
  std::unique_ptr<Math::Vector<float>> m_dbiases{
      std::make_unique<Math::Vector<float>>()};

  std::unique_ptr<Math::Matrix<float>> m_weightCache{
      std::make_unique<Math::Matrix<float>>()};
  std::unique_ptr<Math::Matrix<float>> m_weightMomentums{
      std::make_unique<Math::Matrix<float>>()};
  std::unique_ptr<Math::Vector<float>> m_biasCache{
      std::make_unique<Math::Vector<float>>()};
  std::unique_ptr<Math::Vector<float>> m_biasMomentums{
      std::make_unique<Math::Vector<float>>()};

  float m_l1Weight{};
  float m_l1Bias{};
  float m_l2Weight{};
  float m_l2Bias{};
};
} // namespace Layer
