#pragma once

#include "ann/layers/dense.h"
#include "optimizer.h"

namespace ANN {
namespace Optimizers {
class Adagrad : public Optimizer {
public:
  Adagrad(float learningRate = 1e-2f, float decay = 0.0, float epsilon = 1e-7f)
      : m_startingLearningRate{learningRate}, m_learningRate{learningRate},
        m_decay{decay}, m_epsilon{epsilon} {};

  void preUpdate();

  void updateParams(Layers::Dense &layer) const;

  void postUpdate();

  float learningRate() const { return m_learningRate; }

private:
  float m_startingLearningRate{};
  float m_learningRate{};
  float m_decay{};
  float m_epsilon{};
  float m_iteration{1};
};
} // namespace Optimizers
} // namespace ANN
