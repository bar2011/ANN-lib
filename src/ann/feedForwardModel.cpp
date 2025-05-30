#include "ann/feedForwardModel.h"
#include "ann/exception.h"
#include "ann/layer.h"
#include "ann/loss/categorical.h"
#include "ann/loss/categoricalSoftmax.h"
#include "ann/loss/loss.h"
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

#include "math/random.h"
#include "utils/timer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <variant>

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

void FeedForwardModel::train(const Math::MatrixBase<float> &inputs,
                             const Math::MatrixBase<float> &correct) {
  // If loss is categorical, use more efficient route of converting correct from
  // begin 1-hot encoded to a vector of indicies, and training on them
  if (std::holds_alternative<Loss::Categorical>(*m_loss) ||
      std::holds_alternative<Loss::CategoricalSoftmax>(*m_loss)) {
    auto correctVector{argmaxFloat(correct)};
    train(inputs, *correctVector);
    return;
  }

  Utils::Timer epochTime{};   // Used to track time passed in each epoch
  Utils::Timer displayTime{}; // Used for displaying update messages

  const size_t stepNum{inputs.rows() / m_batchSize};

  // Save std::cout config to later restore it
  const auto coutFlags{std::cout.flags()};
  const auto coutPrecision{std::cout.precision()};

  // Set up floating point printing for training updates
  std::cout << std::fixed << std::setprecision(4);

  for (size_t epoch{}; epoch < m_epochs; ++epoch) {
    if (m_verbose)
      std::cout << "\nEpoch " << epoch + 1 << ":\n";

    // Set up batch sequence
    std::vector<size_t> batchSequence{createBatchSequence(stepNum)};

    epochTime.reset();
    for (size_t batch{}; batch < stepNum; ++batch) {
      const auto batchData{
          inputs.view(batchSequence[batch] * m_batchSize,
                      (batchSequence[batch] + 1) * m_batchSize)};
      const auto batchCorrect{
          correct.view(batchSequence[batch] * m_batchSize,
                       (batchSequence[batch] + 1) * m_batchSize)};

      forward(batchData);

      std::shared_ptr<const Math::Matrix<float>> outputGradients{};
      // Loss forward + backward
      std::visit(
          [&layers = m_layers, &batchCorrect,
           &outputGradients](Loss::Loss &loss) {
            loss.forward(layers[layers.size() - 1]->output(), batchCorrect);
            outputGradients = loss.backward();
          },
          *m_loss);

      optimize(outputGradients);

      // Display information every about half second, or at the first/final
      // batch, only if verbose is true
      if (m_verbose && (displayTime.elapsed() >= 0.5 || batch + 1 == stepNum ||
                        batch == 0)) {
        printUpdate(displayTime.elapsed(), epochTime.elapsed(), batch, stepNum);
        displayTime.reset();
      }
    }
    std::cout << '\n';
  }

  // Restore previous std::cout config
  std::cout.flags(coutFlags);
  std::cout.precision(coutPrecision);
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

std::vector<size_t>
FeedForwardModel::createBatchSequence(size_t stepNum) const {
  std::vector<size_t> batchSequence(stepNum);
  std::iota(batchSequence.begin(), batchSequence.end(), 0);
  if (m_shuffleBatches)
    std::shuffle(batchSequence.begin(), batchSequence.end(), Math::Random::mt);
  return batchSequence;
}

void FeedForwardModel::forward(
    std::shared_ptr<const Math::MatrixBase<float>> batchData) {
  for (size_t i{}; i < m_layers.size(); ++i) {
    auto layerInputs{(i == 0) ? batchData : m_layers[i - 1]->output()};
    m_layers[i]->forward(layerInputs);
  }
}

void FeedForwardModel::optimize(
    std::shared_ptr<const Math::MatrixBase<float>> outputGradients) {
  m_optimizer->preUpdate();
  // i-- in condition because i is size_t, thus will wrap to max if negative
  for (size_t i{m_layers.size()}; i-- > 0;) {
    auto currentDvalues{(i == m_layers.size() - 1)
                            ? outputGradients
                            : m_layers[i + 1]->dinputs()};
    m_layers[i]->backward(currentDvalues);
    if (m_layers[i]->isTrainable())
      switch (m_layers[i]->type()) {
      case Layer::Type::Dense: {
        m_optimizer->updateParams(dynamic_cast<Layers::Dense &>(*m_layers[i]));
        break;
      }
      default:
        break;
      }
  }
  m_optimizer->postUpdate();
}

void FeedForwardModel::printUpdate(double displayTime, double epochTime,
                                   size_t currentBatch, size_t stepNum) const {
  // Calculate loss to be displayed (only if verbose is true)
  float dataLoss{};
  float regularizationLoss{};
  calculateLoss(dataLoss, regularizationLoss);

  // Get accuracy only for classification losses (no regression accuracy)
  std::stringstream accuracy{};
  calculateAccuracy(accuracy);

  std::cout << '\r' << currentBatch + 1 << '/' << stepNum << '\t'
            << static_cast<size_t>(epochTime) << "s "
            << formatTime(epochTime / (currentBatch + 1)) << "/step \t"
            << accuracy.str() << "loss: " << dataLoss + regularizationLoss
            << " (data loss: " << dataLoss
            << ", reg loss: " << regularizationLoss
            << ") - lr: " << m_optimizer->learningRate() << "                 "
            << std::flush;
}

void FeedForwardModel::calculateLoss(float &dataLoss,
                                     float &regularizationLoss) const {
  std::visit(
      [&dataLoss, &regularizationLoss, &layers = m_layers](Loss::Loss &loss) {
        // Calculate data loss
        dataLoss = loss.mean();
        // Sum all regularization losses into the single variable
        // i-- in condition because i is size_t, thus will wrap to max if
        // negative
        for (size_t i{layers.size()}; i-- > 0;) {
          if (layers[i]->isTrainable())
            switch (layers[i]->type()) {
            case Layer::Type::Dense: {
              regularizationLoss += loss.regularizationLoss(
                  dynamic_cast<Layers::Dense &>(*layers[i]));
              break;
            }
            default:
              break;
            }
        }
      },
      *m_loss);
}

void FeedForwardModel::calculateAccuracy(std::stringstream &accuracy) const {
  accuracy << std::fixed << std::setprecision(4);
  std::visit(overloaded{[&accuracy](Loss::Categorical &l) {
                          accuracy << "accuracy: " << l.accuracy() << " - ";
                        },
                        [&accuracy](Loss::CategoricalSoftmax &l) {
                          accuracy << "accuracy: " << l.accuracy() << " - ";
                        },
                        [&accuracy](Loss::Binary &l) {
                          accuracy << "accuracy: " << l.accuracy() << " - ";
                        },
                        [&accuracy](Loss::MSE &l) {},
                        [&accuracy](Loss::MAE &l) {}},
             *m_loss);
}

std::string FeedForwardModel::formatTime(double seconds) {
  if (seconds >= 1.0)
    return std::to_string(static_cast<unsigned long long>(seconds)) + "s";

  const unsigned long long ns = static_cast<unsigned long long>(seconds * 1e9);
  if (ns < 1000)
    return std::to_string(ns) + "ns";

  const unsigned long long us = ns / 1000;
  if (us < 1000)
    return std::to_string(us) + "us";

  const unsigned long long ms = us / 1000;
  if (ms < 1000)
    return std::to_string(ms) + "ms";

  // Fallback: round down to 0s
  return "0s";
}
} // namespace ANN
