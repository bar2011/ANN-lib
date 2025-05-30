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

  // Helper template for std::visit
  // Credit: cppreference.com
  template <class... Ts> struct overloaded : Ts... {
    using Ts::operator()...;
  };

public:
  using LossVariant = std::variant<Loss::Categorical, Loss::CategoricalSoftmax,
                                   Loss::Binary, Loss::MSE, Loss::MAE>;

  FeedForwardModel() = default;

  FeedForwardModel(ModelDesc modelDescriptor);

  FeedForwardModel(ModelDesc modelDescriptor, TrainDesc trainingDescriptor);

  // Loads given model descriptor into configuration
  // Throws if passed in model has an empty layers array
  void configure(ModelDesc modelDescriptor);

  // Loads given training descriptor into configuration
  void configure(TrainDesc trainingDescriptor);

  // Loads given descriptors into configuration
  void configure(ModelDesc modelDescriptor, TrainDesc trainingDescriptor);

  // Train network based on given inputs
  // inputs dims - (X, input_num)
  // correct dims - (X, output_num)
  // X / batch_size = steps per epoch
  // For categorical cross-entropy loss, `correct` should be one=hot encoded
  void train(const Math::MatrixBase<float> &inputs,
             const Math::MatrixBase<float> &correct);

  // Train network based on given inputs
  // correct - vector of correct output index for the corresponding inputs
  // inputs dims - (X, input_num)
  // correct dims - (X)
  // X / batch_size = steps per epoch
  // throws if loss isn't categorical cross-entropy
  void train(const Math::MatrixBase<float> &inputs,
             const Math::VectorBase<float> &correct);

  // Predict single input
  Math::Vector<float> predict(const Math::VectorBase<float> &inputs) const;
  // Predict input batch
  Math::Matrix<float> predict(const Math::MatrixBase<float> &inputs) const;

private:
  // CONFIG FUNCTIONS
  // addLayer overloads (for unpacking LayerDescriptor)
  void addLayer(Dense &, unsigned int &inputs);
  void addLayer(Dropout &, unsigned int &inputs);
  void addLayer(Step &, unsigned int &inputs);
  void addLayer(ReLU &, unsigned int &inputs);
  void addLayer(LeakyReLU &, unsigned int &inputs);
  void addLayer(Sigmoid &, unsigned int &inputs);
  void addLayer(Softmax &, unsigned int &inputs);

  // setLoss overloads (for unpacking TrainingDescriptor
  void setLoss(CategoricalCrossEntropyLoss &);
  void setLoss(CategoricalCrossEntropySoftmaxLoss &);
  void setLoss(BinaryCrossEntropyLoss &);
  void setLoss(MeanSquaredErrorLoss &);
  void setLoss(MeanAbsoluteErrorLoss &);

  // setOptimizer overloads (for unpacking TrainingDescriptor
  void setOptimizer(SGD &);
  void setOptimizer(AdaGrad &);
  void setOptimizer(RMSProp &);
  void setOptimizer(Adam &);

  // TRAINING FUNCTIONS
  std::vector<size_t> createBatchSequence(size_t stepNum) const;
  // Forwards batchData through layers (not loss)
  void forward(std::shared_ptr<const Math::MatrixBase<float>> batchData);
  // Performs backward pass accross all layers, and optimizes trainable layers
  // Inputs - matrix of gradients for the final layer in the network
  void optimize(std::shared_ptr<const Math::MatrixBase<float>> outputGradients);

  // Shows info about current network progression
  void printUpdate(double displayTime, double epochTime, size_t currentBatch,
                   size_t stepNum) const;
  void calculateLoss(float &dataLoss, float &regularizationLoss) const;
  // Get formatted accuracy (if exists). In format for usage by printUpdate()
  void calculateAccuracy(std::stringstream &accuracy) const;

  // Formats given time into string with units - ns, us, ms, or s
  static std::string formatTime(double seconds);

  // Transforms given float one-hot encoded matrix into index matrix
  std::shared_ptr<Math::Vector<float>>
  argmaxFloat(const Math::MatrixBase<float> &m);

  unsigned int m_inputs{};
  std::vector<std::unique_ptr<Layer>> m_layers{};

  std::unique_ptr<LossVariant> m_loss{};
  std::unique_ptr<Optimizers::Optimizer> m_optimizer{};
  size_t m_batchSize{};
  size_t m_epochs{};
  bool m_shuffleBatches{};
  bool m_verbose{};

  // Is model data loaded
  bool m_isModelLoaded{false};
  // Is training data loaded
  bool m_isTrainLoaded{false};
};
} // namespace ANN
