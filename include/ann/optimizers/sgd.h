#pragma once

#include "ann/layers/dense.h"

namespace Optimizers {
class SGD {
public:
  SGD(double learningRate = 1.0, double decay = 0.0, double momentum = 0.0)
      : m_startingLearningRate{learningRate}, m_learningRate{learningRate},
        m_decay{decay}, m_momentum{momentum} {};

  void preUpdate();

  template <typename T> void updateParams(Layer::Dense<T> &layer);

  void postUpdate();

  double learningRate() const { return m_learningRate; }

private:
  double m_startingLearningRate{};
  double m_learningRate{};
  double m_decay{};
  double m_momentum{};
  double m_iteration{1};
};
} // namespace Optimizers

#include "sgd.tpp"
