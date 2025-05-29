#include "ann/feedForwardModel.h"
#include "ann/exception.h"
#include "ann/modelDescriptors.h"

#include "ann/activations/leakyRelu.h"
#include "ann/activations/relu.h"
#include "ann/activations/sigmoid.h"
#include "ann/activations/softmax.h"
#include "ann/activations/step.h"

#include "ann/layers/dense.h"
#include "ann/layers/dropout.h"

#include "ann/optimizers/adagrad.h"
#include "ann/optimizers/adam.h"
#include "ann/optimizers/rmsprop.h"
#include "ann/optimizers/sgd.h"
namespace ANN {
FeedForwardModel::FeedForwardModel(ModelDesc modelDescriptor) {
  configure(modelDescriptor);
}

FeedForwardModel::FeedForwardModel(ModelDesc modelDescriptor,
                                   TrainDesc trainingDescriptor) {
  configure(modelDescriptor, trainingDescriptor);
}

void FeedForwardModel::configure(ModelDesc modelDescriptor) {
  if (modelDescriptor.layers.size() == 0)
    throw ANN::Exception{
        "ANN::FeedForwardModel::configure(FeedForwardModelDescriptor)",
        "Can't configure model with an empty layer array"};

  m_inputs = modelDescriptor.inputs;
  unsigned int currentInputs{m_inputs};
  for (auto &layerVariant : modelDescriptor.layers)
    std::visit(
        [this, &currentInputs](auto &layer) { addLayer(layer, currentInputs); },
        layerVariant);

  // Set that a model was loaded
  m_isModelLoaded = true;
}

void FeedForwardModel::configure(TrainDesc trainingDescriptor) {
  std::visit([this](auto &loss) { setLoss(loss); }, trainingDescriptor.loss);
  std::visit([this](auto &opt) { setOptimizer(opt); },
             trainingDescriptor.optimizer);

  m_batchSize = trainingDescriptor.batchSize;
  m_epochs = trainingDescriptor.epochs;
  m_shuffleBatches = trainingDescriptor.shuffleBatches;
  m_verbose = trainingDescriptor.verbose;

  // Set that training configuration was loaded
  m_isTrainLoaded = true;
}

void FeedForwardModel::configure(ModelDesc modelDescriptor,
                                 TrainDesc trainingDescriptor) {
  configure(modelDescriptor);
  configure(trainingDescriptor);
}

void FeedForwardModel::addLayer(Dense &dense, unsigned int &inputs) {
  m_layers.push_back(std::make_unique<Layers::Dense>(
      inputs, dense.neurons, dense.initMethod, dense.l1Weight, dense.l1Bias,
      dense.l2Weight, dense.l2Bias));
  inputs = dense.neurons; // Update inputs for later layers
}
void FeedForwardModel::addLayer(Dropout &dropout, unsigned int &) {
  m_layers.push_back(std::make_unique<Layers::Dropout>(dropout.dropRate));
}
void FeedForwardModel::addLayer(Step &, unsigned int &) {
  m_layers.push_back(std::make_unique<Activation::Step>());
}
void FeedForwardModel::addLayer(ReLU &, unsigned int &) {
  m_layers.push_back(std::make_unique<Activation::ReLU>());
}
void FeedForwardModel::addLayer(LeakyReLU &lrelu, unsigned int &) {
  m_layers.push_back(std::make_unique<Activation::LeakyReLU>(lrelu.alpha));
}
void FeedForwardModel::addLayer(Sigmoid &, unsigned int &) {
  m_layers.push_back(std::make_unique<Activation::Sigmoid>());
}
void FeedForwardModel::addLayer(Softmax &, unsigned int &) {
  m_layers.push_back(std::make_unique<Activation::Softmax>());
}

void FeedForwardModel::setLoss(CategoricalCrossEntropyLoss &) {
  m_loss = std::make_unique<LossVariant>(Loss::Categorical{});
}
void FeedForwardModel::setLoss(CategoricalCrossEntropySoftmaxLoss &) {
  m_loss = std::make_unique<LossVariant>(Loss::CategoricalSoftmax{});
}
void FeedForwardModel::setLoss(BinaryCrossEntropyLoss &) {
  m_loss = std::make_unique<LossVariant>(Loss::Binary{});
}
void FeedForwardModel::setLoss(MeanSquaredErrorLoss &) {
  m_loss = std::make_unique<LossVariant>(Loss::MSE{});
}
void FeedForwardModel::setLoss(MeanAbsoluteErrorLoss &) {
  m_loss = std::make_unique<LossVariant>(Loss::MAE{});
}

void FeedForwardModel::setOptimizer(SGD &sgd) {
  m_optimizer = std::make_unique<Optimizers::SGD>(sgd.learningRate, sgd.decay,
                                                  sgd.momentum);
}
void FeedForwardModel::setOptimizer(AdaGrad &adagrad) {
  m_optimizer = std::make_unique<Optimizers::Adagrad>(
      adagrad.learningRate, adagrad.decay, adagrad.epsilon);
}
void FeedForwardModel::setOptimizer(RMSProp &rmsprop) {
  m_optimizer = std::make_unique<Optimizers::RMSProp>(
      rmsprop.learningRate, rmsprop.decay, rmsprop.epsilon, rmsprop.rho);
}
void FeedForwardModel::setOptimizer(Adam &adam) {
  m_optimizer = std::make_unique<Optimizers::Adam>(
      adam.learningRate, adam.decay, adam.epsilon, adam.beta1, adam.beta2);
}
} // namespace ANN
