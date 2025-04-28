#include "ann/layers/loss.h"

namespace Layer {
double Loss::mean() {
  double outputSum{};

  for (size_t i{}; i < m_output->size(); ++i)
    outputSum += (*m_output)[i];

  return outputSum / static_cast<double>(m_output->size());
}
} // namespace Layer
