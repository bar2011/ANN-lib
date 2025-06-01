#pragma once

#include "../layer.h"
#include "../modelDescriptors.h"

#include "math/matrix.h"
#include "math/matrixBase.h"
#include "math/vector.h"

// Forward declarations

namespace ANN {
namespace Optimizers {
class Optimizer;
class SGD;
class Adagrad;
class RMSProp;
class Adam;
} // namespace Optimizers

namespace Loss {
class Loss;
}

namespace Layers {
// Basic Dense layer class.
class Dense : public Layer {
public:
  Dense() = delete;

  // 0-init inputs, biases, and outputs
  // Uses activation of linear as default.
  Dense(unsigned int inputNum, unsigned int neuronNum,
        ANN::WeightInit initMethod = ANN::WeightInit::Random,
        float l1Weight = 0, float l1Bias = 0, float l2Weight = 0,
        float l2Bias = 0);

  // Copy constructor deleted
  Dense(const Dense &other) = delete;

  // Move constructor
  Dense(Dense &&other);

  // Copy assignment deleted
  Dense &operator=(const Dense &other) = delete;

  // Move assignment
  Dense &operator=(Dense &&other);

  // Forward pass: stores and returns layer outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  virtual const Math::Matrix<float> &
  forward(const Math::MatrixBase<float> &inputs);

  // Forward pass without storing layer outputs
  // inputs dimensions - (batch_num, input_num)
  // outputs dimensions - (batch_num, neuron_num)
  virtual Math::Matrix<float>
  predict(const Math::MatrixBase<float> &inputs) const;

  // Backward pass: stores parameters gradients and returns input gradients
  // dvalues dimensions - (batch_num, neuron_num)
  // outputs dimensions - (batch_num, input_num)
  virtual const Math::Matrix<float> &
  backward(const Math::MatrixBase<float> &dvalues);

  // Loads weights/biases into layer.
  // Given parameters will be invalid after the function is called
  void loadWeights(Math::Matrix<float> &weights);
  void loadBiases(Math::Vector<float> &biases);

  virtual void saveParams(std::ofstream &file) const;
  virtual void loadParams(std::ifstream &file);

  const Math::Matrix<float> &weights() const { return m_weights; }
  const Math::Vector<float> &biases() const { return m_biases; }
  virtual const Math::Matrix<float> &output() const { return m_output; }
  virtual const Math::Matrix<float> &dinputs() const { return m_dinputs; }

  virtual bool isTrainable() const { return true; }
  virtual std::string_view name() const { return "Dense"; }
  virtual Layer::Type type() const { return Layer::Type::Dense; }

  friend class Optimizers::Optimizer;
  friend class Optimizers::SGD;
  friend class Optimizers::Adagrad;
  friend class Optimizers::RMSProp;
  friend class Optimizers::Adam;

  friend class Loss::Loss;

private:
  Math::MatrixView<float> m_input{};
  Math::Matrix<float> m_weights{};
  Math::Vector<float> m_biases{};
  Math::Matrix<float> m_output{};

  Math::Matrix<float> m_dweights{};
  Math::Matrix<float> m_dinputs{};
  Math::Vector<float> m_dbiases{};

  Math::Matrix<float> m_weightCache{};
  Math::Matrix<float> m_weightMomentums{};
  Math::Vector<float> m_biasCache{};
  Math::Vector<float> m_biasMomentums{};

  float m_l1Weight{};
  float m_l1Bias{};
  float m_l2Weight{};
  float m_l2Bias{};
};
} // namespace Layers
} // namespace ANN
