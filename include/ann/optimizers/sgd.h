#pragma once

#include "ann/layers/dense.h"
#include "optimizer.h"

namespace ANN {
namespace Optimizers {
class SGD : public Optimizer {
public:
  SGD(float learningRate = 1e-2f, float decay = 0.0, float momentum = 0.0)
      : m_startingLearningRate{learningRate}, m_learningRate{learningRate},
        m_decay{decay}, m_momentum{momentum} {};

  void preUpdate();

  void updateParams(Layer::Dense &layer) const;

  void postUpdate();

  float learningRate() const { return m_learningRate; }

private:
  float m_startingLearningRate{};
  float m_learningRate{};
  float m_decay{};
  float m_momentum{};
  float m_iteration{1};
};
} // namespace Optimizers
} // namespace ANN
