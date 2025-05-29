#pragma once

#include "ann/modelDescriptors.h"
#include "ann/optimizers/optimizer.h"
#include "layer.h"

#include "ann/loss/MAE.h"
#include "ann/loss/MSE.h"
#include "ann/loss/binary.h"
#include "ann/loss/categorical.h"
#include "ann/loss/categoricalSoftmax.h"

#include "math/matrixBase.h"
#include "math/vector.h"
#include "math/vectorBase.h"

#include <memory>
#include <vector>

namespace ANN {

class FeedForwardModel {
private:
  using ModelDesc = FeedForwardModelDescriptor;
  using TrainDesc = FeedForwardTrainingDescriptor;

public:
  using LossVariant = std::variant<Loss::Categorical, Loss::CategoricalSoftmax,
                                   Loss::Binary, Loss::MSE, Loss::MAE>;

  FeedForwardModel() = default;

  FeedForwardModel(ModelDesc modelDescriptor);

  FeedForwardModel(ModelDesc modelDescriptor, TrainDesc trainingDescriptor);

  // Loads given descriptor into configuration
  // Throws if passed in model has an empty layers array
  void configure(ModelDesc modelDescriptor);

  void configure(TrainDesc trainingDescriptor);

  void configure(ModelDesc modelDescriptor, TrainDesc trainingDescriptor);

  // Train network based on given inputs
  // inputs dims - (X, input_num)
  // correct dims - (X, output_num)
  // X / batch_num = steps per epoch
  // For categorical cross-entropy loss, throws if 'correct' isn't 1-hot encoded
  void train(const Math::MatrixBase<float> &inputs,
             const Math::MatrixBase<float> &correct);

  // Train network based on given inputs
  // correct - vector of correct output index for the corresponding inputs
  // inputs dims - (X, input_num)
  // correct dims - (X)
  // X / batch_num = steps per epoch
  // throws if loss isn't categorical cross-entropy
  void train(const Math::MatrixBase<float> &inputs,
             const Math::VectorBase<float> &correct);

  // Predict single input
  Math::Vector<float> predict(const Math::VectorBase<float> &inputs) const;
  // Predict input batch
  Math::Matrix<float> predict(const Math::MatrixBase<float> &inputs) const;

private:
  // addLayer overloads (for unpacking LayerDescriptor)
  void addLayer(Dense &, unsigned int &inputs);
  void addLayer(Dropout &, unsigned int &inputs);
  void addLayer(Step &, unsigned int &inputs);
  void addLayer(ReLU &, unsigned int &inputs);
  void addLayer(LeakyReLU &, unsigned int &inputs);
  void addLayer(Sigmoid &, unsigned int &inputs);
  void addLayer(Softmax &, unsigned int &inputs);

  unsigned int m_inputs{};
  std::vector<std::unique_ptr<Layer>> m_layers{};

  size_t m_batchSize{};
  size_t m_epochs{};
  std::unique_ptr<LossVariant> m_loss{};
  std::unique_ptr<Optimizers::Optimizer> m_optimizer{};

  // Is model data loaded
  bool m_isModelLoaded{false};
  // Is training data loaded
  bool m_isTrainLoaded{false};
};
} // namespace ANN
