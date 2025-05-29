#pragma once

#include "../layer.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Layers {
// Dropout activation as a layer
class Dropout : public Layer {
public:
  // dropout - rate of neurons to *disable*
  Dropout(float dropout);

  // Copy constructor deleted
  Dropout(const Dropout &other) = delete;

  // Move constructor
  Dropout(Dropout &&other) noexcept;

  // Copy assignment deleted
  Dropout &operator=(const Dropout &other) = delete;

  // Move assignment
  Dropout &operator=(Dropout &&other) noexcept;

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  virtual std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs);

  // Compute dinputs gradient
  virtual std::shared_ptr<const Math::Matrix<float>>
  backward(const std::shared_ptr<const Math::MatrixBase<float>> &dvalues);

  virtual std::shared_ptr<const Math::Matrix<float>> output() const {
    return m_output;
  }

  virtual std::shared_ptr<const Math::Matrix<float>> dinputs() const {
    return m_dinputs;
  }

  virtual bool isTrainable() const { return false; }

  virtual std::string_view name() const { return "Dropout"; }

  virtual Layer::Type type() const { return Layer::Type::Dropout; }

private:
  std::shared_ptr<Math::Matrix<float>> m_output{
      std::make_shared<Math::Matrix<float>>()};

  std::shared_ptr<Math::Matrix<float>> m_mask{
      std::make_shared<Math::Matrix<float>>()};
  float m_dropout{};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Layers
} // namespace ANN
