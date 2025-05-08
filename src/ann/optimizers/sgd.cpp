#include "ann/optimizers/sgd.h"

namespace Optimizers {
void SGD::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7f);
}

void SGD::updateParams(Layer::Dense &layer) const {
  auto learningRate{m_learningRate};
  auto momentum{m_momentum};

  // update weight/bias update member variables to account for momentum
  layer.m_weightUpdates->transform(
      *layer.m_dweights, [learningRate, momentum](float *w, const float *d) {
        *w = momentum * *w - learningRate * *d;
      });
  layer.m_biasUpdates->transform(
      *layer.m_dbiases, [learningRate, momentum](float *b, const float *d) {
        *b = momentum * *b - learningRate * *d;
      });

  // use weight/bias update to update weights/biases
  layer.m_weights->transform(
      *layer.m_weightUpdates,
      [learningRate](float *w, const float *u) { *w += *u; });
  layer.m_biases->transform(
      *layer.m_biasUpdates,
      [learningRate](float *b, const float *u) { *b += *u; });
}

void SGD::postUpdate() { ++m_iteration; }
} // namespace Optimizers
