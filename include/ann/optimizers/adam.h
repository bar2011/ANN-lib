#pragma once

#include "ann/layers/dense.h"
#include "optimizer.h"

namespace Optimizers {
class Adam : public Optimizer {
public:
  Adam(float learningRate = 1e-3f, float decay = 0.0, float epsilon = 1e-7,
       float beta1 = 0.9f, float beta2 = 0.999)
      : m_startingLearningRate{learningRate}, m_learningRate{learningRate},
        m_decay{decay}, m_epsilon{epsilon}, m_beta1{beta1}, m_beta2{beta2} {};

  void preUpdate();

  void updateParams(Layer::Dense &layer) const;

  void postUpdate();

  float learningRate() const { return m_learningRate; }

private:
  float m_startingLearningRate{};
  float m_learningRate{};
  float m_decay{};
  float m_epsilon{};
  float m_beta1{};
  float m_beta2{};
  float m_iteration{1};
};
} // namespace Optimizers
