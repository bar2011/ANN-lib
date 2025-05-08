#include "ann/layers/loss.h"

namespace Layer {
float Loss::mean() {
  float outputSum{};

  for (size_t i{}; i < m_output->size(); ++i)
    outputSum += (*m_output)[i];

  return outputSum / static_cast<float>(m_output->size());
}
} // namespace Layer
