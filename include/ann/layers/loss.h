#pragma once

#include "math/vector.h"

namespace Layer {
// Base loss class - not instantiatable
class Loss {
public:
  // Calculate average loss from calculated output derived from member function
  double mean();

  std::shared_ptr<const Math::Vector<double>> getOutput() const {
    return m_output;
  }

protected:
  Loss() = default;

  // initialize member variables
  Loss(size_t batchNum) : m_output{new Math::Vector<double>(batchNum)} {};

  // Move constructor
  Loss(Loss &&other) : m_output{other.m_output} { other.m_output.reset(); }

  std::shared_ptr<Math::Vector<double>> m_output{new Math::Vector<double>{}};
};
} // namespace Layer
