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
  for (auto &layer : modelDescriptor.layers)
    std::visit([this, &currentInputs](auto &l) { addLayer(l, currentInputs); },
               layer);

  // Set that a model was loaded
  m_isModelLoaded = true;
}

void FeedForwardModel::configure(TrainDesc trainingDescriptor) {
  
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
} // namespace ANN
