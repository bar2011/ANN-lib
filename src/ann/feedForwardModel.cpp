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
#include "utils/exceptions.h"
#include "utils/timer.h"
#include "utils/variants.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
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
  if (modelDescriptor.layers.empty())
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Can't configure model with an empty layer array"};
  if (modelDescriptor.inputs <= 0)
    throw ANN::Exception{
        CURRENT_FUNCTION,
        "Can't configure model with a non-positive input number " +
            std::to_string(modelDescriptor.inputs)};

  m_inputs = modelDescriptor.inputs;
  unsigned int currentInputs{m_inputs};
  for (auto &layerVariant : modelDescriptor.layers)
    std::visit(Utils::overloaded{[](std::monostate &) {
                                   throw ANN::Exception{
                                       CURRENT_FUNCTION,
                                       "Empty layer provided."};
                                 },
                                 [this, &currentInputs](auto &layer) {
                                   addLayer(layer, currentInputs);
                                 }},
               layerVariant);

  // Set that a model was loaded
  m_isModelLoaded = true;
}

void FeedForwardModel::configure(TrainDesc trainingDescriptor) {
  if (trainingDescriptor.batchSize <= 0 || trainingDescriptor.epochs <= 0)
    throw ANN::Exception{
        CURRENT_FUNCTION,
        "Can't configure model with non-positive batchSize or epoch number"};
  std::visit(Utils::overloaded{[](std::monostate &) {
                                 throw ANN::Exception{CURRENT_FUNCTION,
                                                      "Empty loss provided."};
                               },
                               [this](auto &loss) { setLoss(loss); }},
             trainingDescriptor.loss);
  std::visit(Utils::overloaded{[](std::monostate &) {
                                 throw ANN::Exception{
                                     CURRENT_FUNCTION,
                                     "Empty optimizer provided."};
                               },
                               [this](auto &opt) { setOptimizer(opt); }},
             trainingDescriptor.optimizer);

  m_batchSize = trainingDescriptor.batchSize;
  m_epochs = trainingDescriptor.epochs;
  m_trainValidationRate = trainingDescriptor.trainValidationRate;
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

void FeedForwardModel::saveParams(const std::string &path) const {
  static_assert(sizeof(float) == 4, "Assumes float is 4 bytes.");

  std::ofstream file{path, std::ios::binary};
  if (!file)
    throw ANN::Exception{CURRENT_FUNCTION, "Unable to open file"};

  static constexpr char magicVal[]{"MAGICVALUE"};

  if (!file.write(magicVal, sizeof(magicVal))) // Magic value
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Error on writing magic value to file"};

  for (auto &layer : m_layers)
    layer->saveParams(file);
}

void FeedForwardModel::loadParams(const std::string &path) {
  static_assert(sizeof(float) == 4, "Assumes float is 4 bytes.");

  std::ifstream file{path, std::ios::binary};
  if (!file)
    throw ANN::Exception{CURRENT_FUNCTION, "Unable to open file"};

  static constexpr char magicVal[]{"MAGICVALUE"};

  char readMagicValue[sizeof(magicVal)]{};
  file.read(readMagicValue, sizeof(magicVal));
  if (!file || strcmp(readMagicValue, magicVal) != 0)
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Magic value didn't match. File might be corrupted"};

  for (auto &layer : m_layers)
    layer->loadParams(file);
}

void FeedForwardModel::train(const Math::MatrixBase<float> &inputs,
                             const Math::MatrixBase<float> &correct,
                             const std::string &logPath) {
  if (!m_isModelLoaded || !m_isTrainLoaded)
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Can't train while model isn't fully loaded"};

  std::ofstream logFile{logPath};
  if (!logFile && logPath != "") {
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Unable to open provided log file in " + logPath};
  }

  // If loss is categorical, use more efficient route of converting correct from
  // begin 1-hot encoded to a vector of indices, and training on them
  if (std::holds_alternative<Loss::Categorical>(m_loss) ||
      std::holds_alternative<Loss::CategoricalSoftmax>(m_loss)) {
    auto correctVector{argmaxFloat(correct)};
    train(inputs, correctVector);
    return;
  }

  Utils::Timer epochTime{};   // Used to track time passed in each epoch
  Utils::Timer displayTime{}; // Used for displaying update messages

  const size_t validationNum{static_cast<size_t>(
      std::ceil(inputs.rows() * (1 - m_trainValidationRate)))};

  auto inputsTraining{inputs.view(0, validationNum)};
  auto correctTraining{correct.view(0, validationNum)};

  const size_t stepNum{inputsTraining.rows() / m_batchSize};

  for (size_t epoch{}; epoch < m_epochs; ++epoch) {
    if (m_verbose)
      std::cout << "\nEpoch " << epoch + 1 << ":\n";
    if (logFile.is_open())
      logFile << "\nEPOCH " << epoch + 1 << ":\n";

    // Set up batch sequence
    std::vector<size_t> batchSequence{createBatchSequence(stepNum)};

    epochTime.reset();
    for (size_t batch{}; batch < stepNum; ++batch) {
      const auto batchData{
          inputsTraining.view(batchSequence[batch] * m_batchSize,
                              (batchSequence[batch] + 1) * m_batchSize)};
      const auto batchCorrect{
          correctTraining.view(batchSequence[batch] * m_batchSize,
                               (batchSequence[batch] + 1) * m_batchSize)};

      forward(batchData);

      Math::MatrixView<float> outputGradients{};
      // Loss forward + backward
      std::visit(
          [&layers = m_layers, &batchCorrect,
           &outputGradients](Loss::Loss &loss) {
            loss.forward(layers.back()->output(), batchCorrect);
            outputGradients = loss.backward().view();
          },
          m_loss);

      optimize(outputGradients);

      // Display information every about half second, or at the first/final
      // batch, only if verbose is true
      if (m_verbose && (displayTime.elapsed() >= 0.5 || batch + 1 == stepNum ||
                        batch == 0)) {
        std::cout << '\r'
                  << getUpdateMsg(epochTime.elapsed(), batch, stepNum).str()
                  << std::flush;
        displayTime.reset();
      }

      // If log file is defined, write to it every update message
      if (logFile.is_open())
        logFile << getUpdateMsg(epochTime.elapsed(), batch, stepNum).str()
                << '\n';
    }

    // Perform validation
    if (m_trainValidationRate > 0 && (m_verbose || logFile.is_open())) {
      // forward validation
      auto valInputs{inputs.view(validationNum, inputs.rows())};
      auto valCorrect{correct.view(validationNum, inputs.rows())};
      forward(valInputs, false);
      float valLoss{};
      std::visit(
          [&layers = m_layers, &valCorrect, &valLoss](Loss::Loss &loss) {
            loss.forward(layers.back()->output(), valCorrect);
            valLoss = loss.mean();
          },
          m_loss);

      if (m_verbose)
        std::cout << " - val loss: " << valLoss;
      if (logFile.is_open())
        logFile << "Validation loss: " << valLoss << '\n';

      if (float valAccuracy{calculateAccuracy()}; valAccuracy != -1) {
        if (m_verbose)
          std::cout << " - val accuracy: " << valAccuracy;
        if (logFile.is_open())
          logFile << "Validation accuracy: " << valAccuracy << '\n';
      }
    }

    std::cout << '\n';
  }
}

void FeedForwardModel::train(const Math::MatrixBase<float> &inputs,
                             const Math::VectorBase<float> &correct,
                             const std::string &logPath) {
  if (!m_isModelLoaded || !m_isTrainLoaded)
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Can't train while model isn't fully loaded"};

  std::ofstream logFile{logPath};
  if (!logFile && logPath != "") {
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Unable to open provided log file in " + logPath};
  }

  // If loss isn't categorical, throw exception
  if (!std::holds_alternative<Loss::Categorical>(m_loss) &&
      !std::holds_alternative<Loss::CategoricalSoftmax>(m_loss))
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Can't train on vector when loss isn't categorical"};

  Utils::Timer epochTime{};   // Used to track time passed in each epoch
  Utils::Timer displayTime{}; // Used for displaying update messages

  const size_t validationNum{static_cast<size_t>(
      std::ceil(inputs.rows() * (1 - m_trainValidationRate)))};

  auto inputsTraining{inputs.view(0, validationNum)};
  auto correctTraining{correct.view(0, validationNum)};

  const size_t stepNum{inputsTraining.rows() / m_batchSize};

  for (size_t epoch{}; epoch < m_epochs; ++epoch) {
    if (m_verbose)
      std::cout << "\nEpoch " << epoch + 1 << ":\n";
    if (logFile.is_open())
      logFile << "\nEPOCH " << epoch + 1 << ":\n";

    // Set up batch sequence
    std::vector<size_t> batchSequence{createBatchSequence(stepNum)};

    epochTime.reset();
    for (size_t batch{}; batch < stepNum; ++batch) {
      const auto batchData{
          inputsTraining.view(batchSequence[batch] * m_batchSize,
                              (batchSequence[batch] + 1) * m_batchSize)};
      const auto batchCorrect{
          correct.view(batchSequence[batch] * m_batchSize,
                       (batchSequence[batch] + 1) * m_batchSize)};

      forward(batchData);

      Math::MatrixView<float> outputGradients{};
      // Loss forward + backward
      std::visit(Utils::overloaded{
                     [&layers = m_layers, &batchCorrect,
                      &outputGradients](Loss::Categorical &loss) {
                       loss.forward(layers.back()->output(), batchCorrect);
                       outputGradients = loss.backward().view();
                     },
                     [&layers = m_layers, &batchCorrect,
                      &outputGradients](Loss::CategoricalSoftmax &loss) {
                       loss.forward(layers.back()->output(), batchCorrect);
                       outputGradients = loss.backward().view();
                     },
                     [](auto &) { assert(false); }},
                 m_loss);

      optimize(outputGradients);

      // Display information every about half second, or at the first/final
      // batch, only if verbose is true
      if (m_verbose && (displayTime.elapsed() >= 0.5 || batch + 1 == stepNum ||
                        batch == 0)) {
        std::cout << '\r'
                  << getUpdateMsg(epochTime.elapsed(), batch, stepNum).str()
                  << std::flush;
        displayTime.reset();
      }

      // If log file is defined, write to it every update message
      if (logFile.is_open())
        logFile << getUpdateMsg(epochTime.elapsed(), batch, stepNum).str()
                << '\n';
    }

    // Perform validation
    if (m_trainValidationRate > 0 && (m_verbose || logFile.is_open())) {
      // forward validation
      auto valInputs{inputs.view(validationNum, inputs.rows())};
      auto valCorrect{correct.view(validationNum, inputs.rows())};
      forward(valInputs);
      float valLoss{};
      std::visit(
          Utils::overloaded{[&layers = m_layers, &valCorrect,
                             &valLoss](Loss::Categorical &loss) {
                              loss.forward(layers.back()->output(), valCorrect);
                              valLoss = loss.mean();
                            },
                            [&layers = m_layers, &valCorrect,
                             &valLoss](Loss::CategoricalSoftmax &loss) {
                              loss.forward(layers.back()->output(), valCorrect);
                              valLoss = loss.mean();
                            },
                            [](auto &) { assert(false); }},
          m_loss);

      if (m_verbose)
        std::cout << " - val loss: " << valLoss;
      if (logFile.is_open())
        logFile << "Validation loss: " << valLoss << '\n';

      if (float valAccuracy{calculateAccuracy()}; valAccuracy != -1) {
        if (m_verbose)
          std::cout << " - val accuracy: " << valAccuracy;
        if (logFile.is_open())
          logFile << "Validation accuracy: " << valAccuracy << '\n';
      }
    }

    std::cout << '\n';
  }
}

Math::Vector<float>
FeedForwardModel::evaluate(const Math::MatrixBase<float> &inputs,
                           const Math::MatrixBase<float> &correct) {
  Math::Vector<float> averageLoss{};
  // Layer forward
  forward(inputs.view(), false);
  // Loss forward
  std::visit(
      [&layers = m_layers, &correct, &averageLoss](Loss::Loss &loss) {
        averageLoss = loss.forward(layers.back()->output(), correct.view());
      },
      m_loss);

  return averageLoss;
}

Math::Vector<float>
FeedForwardModel::evaluate(const Math::MatrixBase<float> &inputs,
                           const Math::VectorBase<float> &correct) {
  // If loss isn't categorical, throw exception
  if (!std::holds_alternative<Loss::Categorical>(m_loss) &&
      !std::holds_alternative<Loss::CategoricalSoftmax>(m_loss))
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Can't evalute on vector when loss isn't categorical"};

  Math::Vector<float> averageLoss{};
  // Layer forward
  forward(inputs.view(), false);
  // Loss forward
  std::visit(Utils::overloaded{[&layers = m_layers, &correct,
                                &averageLoss](Loss::Categorical &loss) {
                                 averageLoss = loss.forward(
                                     layers.back()->output(), correct.view());
                               },
                               [&layers = m_layers, &correct,
                                &averageLoss](Loss::CategoricalSoftmax &loss) {
                                 averageLoss = loss.forward(
                                     layers.back()->output(), correct.view());
                               },
                               [](auto &) { assert(false); }},
             m_loss);

  return averageLoss;
}

Math::Vector<float>
FeedForwardModel::predict(const Math::VectorBase<float> &inputs) const {
  Math::Matrix<float> inputsMatrix{1, inputs.size(), inputs.data().data()};
  Math::Vector<float> prediction{predict(inputsMatrix)};
  return prediction;
}

Math::Matrix<float>
FeedForwardModel::predict(const Math::MatrixBase<float> &inputs) const {
  Math::Matrix<float> output{inputs};
  for (size_t i{}; i < m_layers.size(); ++i) {
    // Skip dropout layers
    if (m_layers[i]->type() == Layer::Type::Dropout)
      continue;

    output = m_layers[i]->predict(output);
  }

  if (auto loss = std::get_if<Loss::CategoricalSoftmax>(&m_loss))
    return loss->predictSoftmax(output);
  return output;
}

void FeedForwardModel::calculateLoss(float *dataLoss,
                                     float *regularizationLoss) const {
  std::visit(
      [&dataLoss, &regularizationLoss,
       &layers = m_layers](const Loss::Loss &loss) {
        // Calculate data loss
        if (dataLoss)
          *dataLoss = loss.mean();
        // Sum all regularization losses into the single variable
        // i-- in condition because i is size_t, thus will wrap to max if
        // negative
        if (regularizationLoss) {
          *regularizationLoss = 0;
          for (size_t i{layers.size()}; i-- > 0;) {
            if (layers[i]->isTrainable())
              switch (layers[i]->type()) {
              case Layer::Type::Dense: {
                *regularizationLoss += loss.regularizationLoss(
                    dynamic_cast<Layers::Dense &>(*layers[i]));
                break;
              }
              default:
                break;
              }
          }
        }
      },
      m_loss);
}

float FeedForwardModel::calculateAccuracy() const {
  float accuracy{-1};
  std::visit(
      Utils::overloaded{
          [&accuracy](const Loss::Categorical &l) { accuracy = l.accuracy(); },
          [&accuracy](const Loss::CategoricalSoftmax &l) {
            accuracy = l.accuracy();
          },
          [&accuracy](const Loss::Binary &l) { accuracy = l.accuracy(); },
          [](const auto &) {}},
      m_loss);
  return accuracy;
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
  m_loss = Loss::Categorical{};
}
void FeedForwardModel::setLoss(CategoricalCrossEntropySoftmaxLoss &) {
  m_loss = Loss::CategoricalSoftmax{};
}
void FeedForwardModel::setLoss(BinaryCrossEntropyLoss &) {
  m_loss = Loss::Binary{};
}
void FeedForwardModel::setLoss(MeanSquaredErrorLoss &) { m_loss = Loss::MSE{}; }
void FeedForwardModel::setLoss(MeanAbsoluteErrorLoss &) {
  m_loss = Loss::MAE{};
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

void FeedForwardModel::forward(const Math::MatrixBase<float> &batchData,
                               bool training) {
  auto layerInputs{batchData.view()};
  for (size_t i{}; i < m_layers.size(); ++i) {
    // If not training, skip dropout layers
    if (!training && m_layers[i]->type() == Layer::Type::Dropout)
      continue;

    layerInputs = m_layers[i]->forward(layerInputs).view();
  }
}

void FeedForwardModel::optimize(
    const Math::MatrixBase<float> &outputGradients) {
  m_optimizer->preUpdate();

  auto currentDValues{outputGradients.view()};
  // i-- in condition because i is size_t, thus will wrap to max if negative
  for (size_t i{m_layers.size()}; i-- > 0;) {
    currentDValues = m_layers[i]->backward(currentDValues).view();
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

std::stringstream FeedForwardModel::getUpdateMsg(double epochTime,
                                                 size_t currentBatch,
                                                 size_t stepNum) const {
  std::stringstream out{};
  out << std::fixed << std::setprecision(4) << currentBatch + 1 << '/'
      << stepNum << '\t' << static_cast<size_t>(epochTime) << "s "
      << formatTime(epochTime / (static_cast<double>(currentBatch + 1)))
      << "/step \t";

  // Calculate loss to be displayed (only if verbose is true)
  float dataLoss{};
  float regularizationLoss{};
  calculateLoss(&dataLoss, &regularizationLoss);

  // Get accuracy only for classification losses (no regression accuracy)
  if (float val{calculateAccuracy()}; val != -1)
    out << "accuracy: " << val << " - ";

  out << "loss: " << dataLoss + regularizationLoss
      << " (data loss: " << dataLoss << ", reg loss: " << regularizationLoss
      << ") - lr: " << m_optimizer->learningRate();

  return out;
}

Math::Vector<float>
FeedForwardModel::argmaxFloat(const Math::MatrixBase<float> &m) {
  // Make a vector of the indices of the biggest values in each row
  // i.e. the correct index in each batch
  Math::Vector<float> max{m.rows()};

  for (size_t i{}; i < m.rows(); ++i)
    for (size_t j{}; j < m.cols(); ++j)
      if (m[i, static_cast<size_t>(max[i])] < m[i, j])
        max[i] = static_cast<float>(j);

  return max;
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
