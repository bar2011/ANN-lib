#pragma once

#include <variant>
#include <vector>

namespace ANN {
enum class WeightInit {
  Xavier,
  He,
  Random,
};

struct Dense {
  unsigned int neurons{};
  WeightInit initMethod{WeightInit::Random};
  float l1Weight{};
  float l1Bias{};
  float l2Weight{};
  float l2Bias{};

  unsigned int inputs{}; // Used internally. No need to define.
};

// Dropout layer descriptor
// dropRate ∈ [0.0, 1.0]
struct Dropout {
  float dropRate{};
};

struct Step {};

struct Sigmoid {};

struct ReLU {};

struct LeakyReLU {
  float alpha{1e-2f};
};

struct Softmax {};

using LayerDescriptor =
    std::variant<Dense, Dropout, Step, Sigmoid, ReLU, LeakyReLU, Softmax>;

struct FeedForwardModelDescriptor {
  unsigned int inputs{};
  std::vector<LayerDescriptor> layers{};
};

struct CategoricalCrossEntropyLoss {};

struct CategoricalCrossEntropySoftmaxLoss {};

struct BinaryCrossEntropyLoss {};

struct MeanSquaredErrorLoss {};

struct MeanAbsoluteErrorLoss {};

using LossDescriptor =
    std::variant<CategoricalCrossEntropyLoss,
                 CategoricalCrossEntropySoftmaxLoss, BinaryCrossEntropyLoss,
                 MeanSquaredErrorLoss, MeanAbsoluteErrorLoss>;

struct SGD {
  float learningRate{1e-2f};
  float decay{};
  float momentum{};
};

struct AdaGrad {
  float learningRate{1e-2f};
  float decay{};
  float epsilon{1e-7f};
};

struct RMSProp {
  float learningRate{1e-3f};
  float decay{};
  float epsilon{1e-7f};
  float rho{0.9f};
};

struct Adam {
  float learningRate{1e-3f};
  float decay{};
  float epsilon{1e-7f};
  float beta1{0.9f};
  float beta2{0.999f};
};

using OptimizerDescriptor = std::variant<SGD, AdaGrad, RMSProp, Adam>;

struct FeedForwardTrainingDescriptor {
  LossDescriptor loss{};
  OptimizerDescriptor optimizer{SGD{}};
  size_t batchSize{32};
  size_t epochs{10};
  bool shuffleBatches{true};
  bool verbose{true};
};
} // namespace ANN
