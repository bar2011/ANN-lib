#pragma once

#include "ann/layers/dense.h"
#include "optimizer.h"

namespace Optimizers {
class RMSProp : public Optimizer {
public:
  RMSProp(float learningRate = 1e-3f, float decay = 0.0, float epsilon = 1e-7f,
          float rho = 0.9f)
      : m_startingLearningRate{learningRate}, m_learningRate{learningRate},
        m_decay{decay}, m_epsilon{epsilon}, m_rho{rho} {};

  void preUpdate();

  void updateParams(Layer::Dense &layer) const;

  void postUpdate();

  float learningRate() const { return m_learningRate; }

private:
  float m_startingLearningRate{};
  float m_learningRate{};
  float m_decay{};
  float m_epsilon{};
  float m_rho{};
  float m_iteration{1};
};
} // namespace Optimizers
