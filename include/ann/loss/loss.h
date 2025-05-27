#pragma once

#include "math/vector.h"

#include <memory>

// Forward declarations
namespace Layer {
class Dense;
}

namespace Loss {
// Base loss class - not instantiatable
class Loss {
public:
  // Calculate average loss from calculated output derived from member function
  float mean();

  std::shared_ptr<const Math::Vector<float>> getOutput() const {
    return m_output;
  }

  float regularizationLoss(const Layer::Dense &layer) const;

protected:
  Loss() = default;

  // Move constructor
  Loss(Loss &&other) : m_output{other.m_output} { other.m_output.reset(); }

  std::shared_ptr<Math::Vector<float>> m_output{
      std::make_shared<Math::Vector<float>>()};
};
} // namespace Loss
