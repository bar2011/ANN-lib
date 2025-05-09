#include "ann/optimizers/adam.h"

#include <cmath>

namespace Optimizers {
void Adam::preUpdate() {
  m_learningRate =
      std::max(m_startingLearningRate / (1 + m_decay * m_iteration), 1e-7f);
}

void Adam::updateParams(Layer::Dense &layer) const {
  auto learningRate{m_learningRate};
  auto epsilon{m_epsilon};
  auto beta1{m_beta1};
  auto beta2{m_beta2};
  auto iteration{m_iteration};
  auto momentumCorrection{1 - std::pow(beta1, iteration + 1)};
  auto cacheCorrection{1 - std::pow(beta2, iteration + 1)};

  // Calculate weight/bias update momentums
  layer.m_weightMomentums->transform(
      *layer.m_dweights,
      [learningRate, beta1](float *weightMomentums, const float *gradient) {
        *weightMomentums = beta1 * *weightMomentums + (1 - beta1) * *gradient;
      });
  layer.m_biasMomentums->transform(
      *layer.m_dbiases,
      [learningRate, beta1](float *biasMomentums, const float *gradient) {
        *biasMomentums = beta1 * *biasMomentums + (1 - beta1) * *gradient;
      });

  // update weight/bias cache member variables to account for adaptive lr
  layer.m_weightCache->transform(
      *layer.m_dweights, [beta2](float *weightCache, const float *gradient) {
        *weightCache =
            beta2 * *weightCache + (1 - beta2) * *gradient * *gradient;
      });
  layer.m_biasCache->transform(
      *layer.m_dbiases, [beta2](float *biasCache, const float *gradient) {
        *biasCache = beta2 * *biasCache + (1 - beta2) * *gradient * *gradient;
      });

  // use weight/bias update to update weights/biases
  layer.m_weights->transform(
      *layer.m_weightMomentums, *layer.m_weightCache,
      [momentumCorrection, cacheCorrection, epsilon, iteration,
       learningRate](float *weight, const float *weightMomentum,
                     const float *weightCache) {
        float correctWeightMomentum{*weightMomentum / momentumCorrection};
        float correctWeightCache{*weightCache / cacheCorrection};
        *weight -= learningRate * correctWeightMomentum /
                   (std::sqrt(correctWeightCache) + epsilon);
      });
  layer.m_biases->transform(
      *layer.m_biasMomentums, *layer.m_biasCache,
      [momentumCorrection, cacheCorrection, epsilon, iteration, learningRate](
          float *bias, const float *biasMomentum, const float *biasCache) {
        float correctBiasMomentum{*biasMomentum / momentumCorrection};
        float correctBiasCache{*biasCache / cacheCorrection};
        *bias -= learningRate * correctBiasMomentum /
                 (std::sqrt(correctBiasCache) + epsilon);
      });
}

void Adam::postUpdate() { ++m_iteration; }
} // namespace Optimizers
