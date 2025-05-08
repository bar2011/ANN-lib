#include "ann/optimizers/adagrad.h"

#include <cmath>

namespace Optimizers {
void Adagrad::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7f);
}

void Adagrad::updateParams(Layer::Dense &layer) const {
  auto learningRate{m_learningRate};
  auto epsilon{m_epsilon};

  // update weight/bias update member variables to account for momentum
  layer.m_weightUpdates->transform(
      *layer.m_dweights, [](float *w, const float *d) { *w += *d * *d; });
  layer.m_biasUpdates->transform(
      *layer.m_dbiases, [](float *b, const float *d) { *b += *d * *d; });

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

void Adagrad::postUpdate() { ++m_iteration; }
} // namespace Optimizers
