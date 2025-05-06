#include "ann/optimizers/sgd.h"

namespace Optimizers {
void SGD::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7);
}

void SGD::postUpdate() { ++m_iteration; }
} // namespace Optimizers
