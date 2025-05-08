#include "ann/optimizers/rmsprop.h"

#include <cmath>

namespace Optimizers {
void RMSProp::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7f);
}

void RMSProp::updateParams(Layer::Dense &layer) const {
  auto learningRate{m_learningRate};
  auto epsilon{m_epsilon};
  auto rho{m_rho};

  // update weight/bias update member variables to account for momentum
  layer.m_weightUpdates->transform(
      *layer.m_dweights, [rho](float *weight, const float *gradient) {
        *weight = rho * *weight + (1 - rho) * *gradient * *gradient;
      });
  layer.m_biasUpdates->transform(
      *layer.m_dbiases, [rho](float *bias, const float *gradient) {
        *bias = rho * *bias + (1 - rho) * *gradient * *gradient;
      });

  // use weight/bias update to update weights/biases
  layer.m_weights->transform(*layer.m_dweights, *layer.m_weightUpdates,
                             [learningRate, epsilon](float *weight,
                                                     const float *gradient,
                                                     const float *updateCache) {
                               *weight -= learningRate * *gradient /
                                          (std::sqrt(*updateCache) + epsilon);
                             });
  layer.m_biases->transform(
      *layer.m_dbiases, *layer.m_biasUpdates,
      [learningRate, epsilon](float *bias, const float *gradient,
                              const float *updateCache) {
        *bias -= learningRate * *gradient / (std::sqrt(*updateCache) + epsilon);
      });
}

void RMSProp::postUpdate() { ++m_iteration; }
} // namespace Optimizers
