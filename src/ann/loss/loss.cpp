#include "ann/loss/loss.h"

#include "ann/layers/dense.h"

#include <numeric>

namespace ANN {
namespace Loss {
float Loss::mean() const {
  float outputSum{};

  for (size_t i{}; i < m_output->size(); ++i)
    outputSum += (*m_output)[i];

  return outputSum / static_cast<float>(m_output->size());
}

float Loss::regularizationLoss(const Layers::Dense &layer) const {
  float regularization{};

  if (layer.m_l1Weight > 0)
    regularization +=
        layer.m_l1Weight *
        std::accumulate(layer.m_weights->data().begin(),
                        layer.m_weights->data().end(), 0.0f,
                        [](float acc, float i) { return acc + std::abs(i); });

  if (layer.m_l1Bias > 0)
    regularization +=
        layer.m_l1Bias *
        std::accumulate(layer.m_biases->data().begin(),
                        layer.m_biases->data().end(), 0.0f,
                        [](float acc, float i) { return acc + std::abs(i); });

  if (layer.m_l2Weight > 0)
    regularization +=
        layer.m_l2Weight *
        std::accumulate(layer.m_weights->data().begin(),
                        layer.m_weights->data().end(), 0.0f,
                        [](float acc, float i) { return acc + i * i; });

  if (layer.m_l2Bias > 0)
    regularization +=
        layer.m_l2Bias *
        std::accumulate(layer.m_biases->data().begin(),
                        layer.m_biases->data().end(), 0.0f,
                        [](float acc, float i) { return acc + i * i; });

  return regularization;
}
} // namespace Loss
} // namespace ANN
