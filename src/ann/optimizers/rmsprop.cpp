#include "ann/optimizers/rmsprop.h"

#include <cmath>

namespace ANN {
namespace Optimizers {
void RMSProp::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7f);
}

void RMSProp::updateParams(Layers::Dense &layer) const {
  auto learningRate{m_learningRate};
  auto epsilon{m_epsilon};
  auto rho{m_rho};
  // update weight/bias cache member variables to account for adaptive lr
  layer.m_weightCache.transform(
      layer.m_dweights, [rho](float *weightCache, const float *gradient) {
        *weightCache = rho * *weightCache + (1 - rho) * *gradient * *gradient;
      });
  layer.m_biasCache.transform(
      layer.m_dbiases, [rho](float *biasCache, const float *gradient) {
        *biasCache = rho * *biasCache + (1 - rho) * *gradient * *gradient;
      });

  // Calculate actual weight/bias updates
  layer.m_weightMomentums.transform(
      layer.m_dweights, layer.m_weightCache,
      [learningRate, epsilon](float *weightUpdates, const float *gradient,
                              const float *cache) {
        *weightUpdates =
            learningRate * *gradient / (std::sqrt(*cache) + epsilon);
      });
  layer.m_biasMomentums.transform(
      layer.m_dbiases, layer.m_biasCache,
      [learningRate, epsilon](float *biasUpdates, const float *gradient,
                              const float *cache) {
        *biasUpdates = learningRate * *gradient / (std::sqrt(*cache) + epsilon);
      });

  // use weight/bias update to update weights/biases
  layer.m_weights.transform(layer.m_weightMomentums,
                            [](float *weight, const float *weightUpdates) {
                              *weight -= *weightUpdates;
                            });
  layer.m_biases.transform(
      layer.m_biasMomentums,
      [](float *bias, const float *biasUpdates) { *bias -= *biasUpdates; });
}

void RMSProp::postUpdate() { ++m_iteration; }
} // namespace Optimizers
} // namespace ANN
