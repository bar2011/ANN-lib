#include "ann/optimizers/sgd.h"

namespace ANN {
namespace Optimizers {
void SGD::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7f);
}

void SGD::updateParams(Layers::Dense &layer) const {
  auto learningRate{m_learningRate};
  auto momentum{m_momentum};

  // update weight/bias momentum member variables
  layer.m_weightMomentums.transform(
      layer.m_dweights, [learningRate, momentum](float *w, const float *d) {
        *w = momentum * *w - learningRate * *d;
      });
  layer.m_biasMomentums.transform(
      layer.m_dbiases, [learningRate, momentum](float *b, const float *d) {
        *b = momentum * *b - learningRate * *d;
      });

  // use weight/bias momentum to update weights/biases
  layer.m_weights.transform(
      layer.m_weightMomentums,
      [learningRate](float *w, const float *u) { *w += *u; });
  layer.m_biases.transform(
      layer.m_biasMomentums,
      [learningRate](float *b, const float *u) { *b += *u; });
}

void SGD::postUpdate() { ++m_iteration; }
} // namespace Optimizers
} // namespace ANN
