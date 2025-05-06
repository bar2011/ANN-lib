#pragma once

#include "sgd.h"

namespace Optimizers {
template <typename T> void SGD::updateParams(Layer::Dense<T> &layer) {
  auto learningRate{m_learningRate};
  auto momentum{m_momentum};

  // update weight/bias update member variables to account for momentum
  layer.m_weightUpdates->transform(
      *layer.m_dweights, [learningRate, momentum](double *w, const double *d) {
        *w = momentum * *w - learningRate * *d;
      });
  layer.m_biasUpdates->transform(
      *layer.m_dbiases, [learningRate, momentum](double *b, const double *d) {
        *b = momentum * *b - learningRate * *d;
      });

  // use weight/bias update to update weights/biases
  layer.m_weights->transform(
      *layer.m_weightUpdates,
      [learningRate](double *w, const double *u) { *w += *u; });
  layer.m_biases->transform(
      *layer.m_biasUpdates,
      [learningRate](double *b, const double *u) { *b += *u; });
}
} // namespace Optimizers
