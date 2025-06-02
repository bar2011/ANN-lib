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
};

// Dropout layer descriptor
// dropRate ∈ <0.0, 1.0>
struct Dropout {
  float dropRate{};
};

struct Step {};

struct Sigmoid {};

struct ReLU {};

struct LeakyReLU {
  float alpha{};
};

struct Softmax {};

using LayerDescriptor = std::variant<std::monostate, Dense, Dropout, Step,
                                     Sigmoid, ReLU, LeakyReLU, Softmax>;

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
    std::variant<std::monostate, CategoricalCrossEntropyLoss,
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

using OptimizerDescriptor =
    std::variant<std::monostate, SGD, AdaGrad, RMSProp, Adam>;

struct FeedForwardTrainingDescriptor {
  LossDescriptor loss{};
  OptimizerDescriptor optimizer{};
  size_t batchSize{32};
  size_t epochs{10};
  // How much of the training data will be reserved to validation
  float trainValidationRate{0.05f};
  bool shuffleBatches{true};
  // If true, prints update messages every ~0.5s
  bool verbose{true};
};
} // namespace ANN
