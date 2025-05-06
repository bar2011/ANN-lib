#pragma once

#include "sgd.h"

namespace Optimizers {
template <typename T> void SGD::updateParams(Layer::Dense<T> &layer) {
  auto learningRate{m_learningRate};
  layer.m_weights->transform(
      *layer.m_dweights,
      [learningRate](double *w, const double *d) { *w -= *d * learningRate; });
  layer.m_biases->transform(
      *layer.m_dbiases,
      [learningRate](double *b, const double *d) { *b -= *d * learningRate; });
}
} // namespace Optimizers
