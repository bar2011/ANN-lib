#pragma once

#include "activation.h"

#include "math/matrix.h"
#include "math/matrixBase.h"

namespace ANN {
namespace Activation {

// Step activation function
class Step : public Activation {
public:
  Step() = default;

  // Copy constructor deleted
  Step(const Step &other) = delete;

  // Move constructor
  Step(Step &&other) noexcept;

  // Copy assignment deleted
  Step &operator=(const Step &other) = delete;

  // Move assignment
  Step &operator=(Step &&other) noexcept;

  // perform forward pass with given batch
  // saves inputs and outputs in member variables
  virtual std::shared_ptr<const Math::Matrix<float>>
  forward(const std::shared_ptr<const Math::MatrixBase<float>> &inputs);

  virtual std::unique_ptr<Math::Matrix<float>>
  predict(const std::shared_ptr<const Math::MatrixBase<float>> &inputs) const;

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

  virtual std::string_view name() const { return "Step"; }

  virtual Layer::Type type() const { return Layer::Type::Step; }

private:
  // No ownership of m_input by the class. Only a view.
  std::shared_ptr<const Math::MatrixBase<float>> m_input{};
  std::shared_ptr<Math::Matrix<float>> m_output{
      std::make_shared<Math::Matrix<float>>()};

  std::shared_ptr<Math::Matrix<float>> m_dinputs{
      std::make_shared<Math::Matrix<float>>()};
};
} // namespace Activation
} // namespace ANN
