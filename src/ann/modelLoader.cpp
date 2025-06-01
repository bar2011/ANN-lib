#include "ann/modelLoader.h"
#include "ann/exception.h"
#include "ann/feedForwardModel.h"
#include "ann/modelDescriptors.h"
#include "utils/exceptions.h"
#include "utils/variants.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <variant>

namespace ANN {
FeedForwardModel ModelLoader::loadFeedForward(const std::string &path) {
  std::ifstream file{path};
  if (!file.is_open())
    throw ANN::Exception{CURRENT_FUNCTION, "Unable to open file " + path};

  FeedForwardModelDescriptor modelDesc{};
  FeedForwardTrainingDescriptor trainDesc{};

  // Uses to identify what to do
  // 0 - reading nothing.
  // 1 - reading model descriptor.
  // 2 - reading training descriptor
  unsigned short mode{0};

  std::string line{};
  unsigned int lineNum{0};
  while (std::getline(file, line)) {
    ++lineNum;
    std::string lineNumStr{std::to_string(lineNum)};
    // Trim whitespace and comments
    trim(line);
    if (line.empty())
      continue;

    // Get current mode. If mode wasn't given, throw exception.
    if (line == "[MODEL]") {
      mode = 1;
      continue;
    } else if (line == "[TRAINING]") {
      mode = 2;
      continue;
    } else if (mode == 0)
      throw ANN::Exception{CURRENT_FUNCTION,
                           "Expected [MODEL] or [TRAINING]. From line " +
                               lineNumStr};

    std::string key, val;
    split(line, '=', key, val, lineNumStr);

    if (mode == 1) {
      // Handle inputs=...
      if (key == "inputs") {
        int inputs{parseStrictInt(val, lineNumStr)};
        if (inputs <= 0)
          throw ANN::Exception{CURRENT_FUNCTION,
                               "Input number needs to be > 0. From line " +
                                   lineNumStr};
        modelDesc.inputs = inputs;
        continue;
      }

      // Handle layers...=...
      if (key.starts_with("layers.")) {
        // Erase "layer." from key start
        key.erase(key.begin(), key.begin() + 7);

        std::string layerNumStr, config;
        split(key, '.', layerNumStr, config, lineNumStr);

        int layerNum{parseStrictInt(layerNumStr, lineNumStr)};

        // Throw if layerNum is bigger by at least two then current layer number
        if (layerNum > modelDesc.layers.size() + 1)
          throw ANN::Exception{CURRENT_FUNCTION,
                               "Can't define layer " + layerNumStr +
                                   " when there has been " +
                                   std::to_string(modelDesc.layers.size()) +
                                   " layers defined. From line " + lineNumStr};

        // Throw if the first occurance of a layer number isn't with "type"
        if (layerNum == modelDesc.layers.size() + 1 && config != "type")
          throw ANN::Exception{CURRENT_FUNCTION,
                               "Unknown layer type. Please declare it before "
                               "any other configuration. From line " +
                                   lineNumStr};

        // Throw if an already existing layer's type has been changed
        if (layerNum <= modelDesc.layers.size() && config == "type")
          throw ANN::Exception{CURRENT_FUNCTION,
                               "A layer's type cannot be changed after its "
                               "initial setting. From line " +
                                   lineNumStr};

        // If all tests have been passed, and config == "type", a new layer can
        // be safely added
        if (config == "type") {
          if (val == "dense")
            modelDesc.layers.push_back(Dense{});
          else if (val == "dropout")
            modelDesc.layers.push_back(Dropout{});
          else if (val == "step")
            modelDesc.layers.push_back(Step{});
          else if (val == "sigmoid")
            modelDesc.layers.push_back(Sigmoid{});
          else if (val == "relu")
            modelDesc.layers.push_back(ReLU{});
          else if (val == "leaky_relu")
            modelDesc.layers.push_back(LeakyReLU{});
          else if (val == "softmax")
            modelDesc.layers.push_back(Softmax{});
          else
            throw ANN::Exception{
                CURRENT_FUNCTION,
                "Unknown layer type provided '" + val +
                    "'. Supported types are: 'dense', 'dropout', 'step', "
                    "'sigmoid', 'relu', 'leaky_relu', 'softmax'. From line " +
                    lineNumStr};
          continue;
        }

        // Safely change a layer's configuration based on layerNum
        std::visit(Utils::overloaded{[](std::monostate &) { assert(false); },
                                     [&config, &val, &lineNumStr](auto &layer) {
                                       configLayer(layer, config, val,
                                                   lineNumStr);
                                     }},
                   modelDesc.layers[layerNum - 1]);

        continue;
      }

      throw ANN::Exception{CURRENT_FUNCTION,
                           "Expected 'inputs' or 'layers...'. From line " +
                               lineNumStr};
    } else if (mode == 2) {
      if (key.starts_with("loss.")) {
        // Erase "loss." from key start
        key.erase(key.begin(), key.begin() + 5);

        if (key == "type") {
          if (val == "categorical_cross_entropy")
            trainDesc.loss = CategoricalCrossEntropyLoss{};
          else if (val == "categorical_cross_entropy_softmax")
            trainDesc.loss = CategoricalCrossEntropySoftmaxLoss{};
          else if (val == "binary_cross_entropy")
            trainDesc.loss = BinaryCrossEntropyLoss{};
          else if (val == "mean_squared_error")
            trainDesc.loss = MeanSquaredErrorLoss{};
          else if (val == "mean_absolute_error")
            trainDesc.loss = MeanAbsoluteErrorLoss{};
          else
            throw ANN::Exception{
                CURRENT_FUNCTION,
                "Unknown loss type provided '" + val +
                    "'. Supported type are: 'categorical_cross_entropy', "
                    "'categorical_cross_entropy_softmax', "
                    "'binary_cross_entropy', 'mean_squared_error', "
                    "'mean_absolute_error'. From line " +
                    lineNumStr};
          continue;
        }

        throw ANN::Exception{
            CURRENT_FUNCTION,
            "Unknown loss configuration provided '" + key +
                "'. Supported configurations are: 'type'. From line " +
                lineNumStr};
        continue;
      }
      if (key.starts_with("optimizer.")) {
        // Erase "optimizer." from key start
        key.erase(key.begin(), key.begin() + 10);

        if (key == "type") {
          if (val == "sgd")
            trainDesc.optimizer = SGD{};
          else if (val == "adagrad")
            trainDesc.optimizer = AdaGrad{};
          else if (val == "rmsprop")
            trainDesc.optimizer = RMSProp{};
          else if (val == "adam")
            trainDesc.optimizer = Adam{};
          else
            throw ANN::Exception{CURRENT_FUNCTION,
                                 "Unknown optimizer type provided '" + val +
                                     "'. Supported type are: 'sgd', 'adagrad', "
                                     "'rmsprop', 'adam'. From line " +
                                     lineNumStr};
          continue;
        }

        std::visit(Utils::overloaded{
                       [&lineNumStr](std::monostate &) {
                         throw ANN::Exception{
                             CURRENT_FUNCTION,
                             "optimizer.type must be declared before any "
                             "further settings. From line " +
                                 lineNumStr};
                       },
                       [&key, &val, &lineNumStr](auto &optimizer) {
                         configOptimizer(optimizer, key, val, lineNumStr);
                       }},
                   trainDesc.optimizer);

        continue;
      }
      if (key == "batch_size") {
        trainDesc.batchSize = parseStrictInt(val, lineNumStr);
        if (trainDesc.batchSize <= 0)
          throw ANN::Exception{CURRENT_FUNCTION,
                               "Batch size must be a natural number "
                               "(integer greater then 0). From line " +
                                   lineNumStr};
        continue;
      }
      if (key == "epochs") {
        trainDesc.epochs = parseStrictInt(val, lineNumStr);
        if (trainDesc.epochs < 0)
          throw ANN::Exception{
              CURRENT_FUNCTION,
              "Epochs must be a non-negative integer. From line " + lineNumStr};
        continue;
      }
      if (key == "train_validation_rate") {
        trainDesc.trainValidationRate = parseStrictFloat(val, lineNumStr);
        if (trainDesc.trainValidationRate <= 0 ||
            trainDesc.trainValidationRate >= 1)
          throw ANN::Exception{CURRENT_FUNCTION,
                               "train_validation_rate must be between "
                               "0 and 1 (not including). From line " +
                                   lineNumStr};
        continue;
      }
      if (key == "shuffle_batches") {
        trainDesc.shuffleBatches = parseStrictBool(val, lineNumStr);
        continue;
      }
      if (key == "verbose") {
        trainDesc.verbose = parseStrictBool(val, lineNumStr);
        continue;
      }

      throw ANN::Exception{CURRENT_FUNCTION,
                           "Expected 'loss...', 'optimizer...', 'batch_size', "
                           "'epochs', 'train_validation_rate', "
                           "'shuffle_batches', or 'verbose'. From line " +
                               lineNumStr};
    }
  }

  // Check if each layer's required config has been dealt with
  for (size_t i{}; i < modelDesc.layers.size(); ++i) {
    auto &layer{modelDesc.layers[i]};
    if (std::holds_alternative<Dense>(layer)) {
      Dense &dense{std::get<Dense>(layer)};
      if (dense.neurons == 0)
        throw ANN::Exception{
            CURRENT_FUNCTION,
            "Required configuration 'neurons' in layer number " +
                std::to_string(i + 1) + " has not been set."};
    } else if (std::holds_alternative<Dropout>(layer)) {
      Dropout &dropout{std::get<Dropout>(layer)};
      if (dropout.dropRate == 0)
        throw ANN::Exception{
            CURRENT_FUNCTION,
            "Required configuration 'drop_rate' in layer number " +
                std::to_string(i + 1) + " has not been set."};
    } else if (std::holds_alternative<LeakyReLU>(layer)) {
      LeakyReLU &leakyReLU{std::get<LeakyReLU>(layer)};
      if (leakyReLU.alpha == 0)
        throw ANN::Exception{CURRENT_FUNCTION,
                             "Required configuration 'alpha' in layer number " +
                                 std::to_string(i + 1) + " has not been set."};
    }
  }

  // Check if input number has been set
  if (modelDesc.inputs == 0)
    throw ANN::Exception{
        CURRENT_FUNCTION,
        "Required model configuration 'inputs' has not been set."};

  return FeedForwardModel{modelDesc, trainDesc};
}

void ModelLoader::configLayer(Dense &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  if (config == "neurons") {
    int neurons{parseStrictInt(value, lineNumStr)};
    if (neurons <= 0)
      throw ANN::Exception{CURRENT_FUNCTION,
                           "Dense layer neuron num must be a natual number "
                           "(integer greater then 0). From line " +
                               lineNumStr};
    layer.neurons = neurons;
    return;
  } else if (config == "init_method") {
    if (value == "random")
      layer.initMethod = WeightInit::Random;
    else if (value == "he")
      layer.initMethod = WeightInit::He;
    else if (value == "xavier")
      layer.initMethod = WeightInit::Xavier;
    else
      throw ANN::Exception{CURRENT_FUNCTION,
                           "Unknown weight initalizer. Allowed initializers "
                           "are: 'random', 'he', and 'xavier'. From line " +
                               lineNumStr};
    return;
  } else if (config == "l1_weight") {
    layer.l1Weight = parseStrictFloat(value, lineNumStr);
    return;
  } else if (config == "l1_bias") {
    layer.l1Bias = parseStrictFloat(value, lineNumStr);
    return;
  } else if (config == "l2_weight") {
    layer.l2Weight = parseStrictFloat(value, lineNumStr);
    return;
  } else if (config == "l2_bias") {
    layer.l2Bias = parseStrictFloat(value, lineNumStr);
    return;
  }
  throw ANN::Exception{
      CURRENT_FUNCTION,
      "Unknown dense configuration provided '" + config +
          "'. Allowed configurations "
          "are: 'neurons', 'init_method', 'l1_weight', 'l1_bias', "
          "'l2_weight', 'l2_bias'. From line " +
          lineNumStr};
}
void ModelLoader::configLayer(Dropout &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  if (config == "drop_rate") {
    layer.dropRate = parseStrictFloat(value, lineNumStr);
    if (layer.dropRate < 0 || layer.dropRate > 1)
      throw ANN::Exception{CURRENT_FUNCTION,
                           "Invalid drop rate provided. Needs to be "
                           "between 0 and 1 (including). From line " +
                               lineNumStr};
    return;
  }
  throw ANN::Exception{
      CURRENT_FUNCTION,
      "Unknown dropout configuration provided '" + config +
          "'. Allowed configurations are: 'drop_rate'. From line " +
          lineNumStr};
}
void ModelLoader::configLayer(Step &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  throw ANN::Exception{CURRENT_FUNCTION,
                       "No step configuration supported. From line " +
                           lineNumStr};
}
void ModelLoader::configLayer(Sigmoid &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  throw ANN::Exception{CURRENT_FUNCTION,
                       "No sigmoid configuration supported. From line " +
                           lineNumStr};
}
void ModelLoader::configLayer(ReLU &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  throw ANN::Exception{CURRENT_FUNCTION,
                       "No relu configuration supported. From line " +
                           lineNumStr};
}
void ModelLoader::configLayer(LeakyReLU &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  if (config == "alpha") {
    layer.alpha = parseStrictFloat(value, lineNumStr);
    return;
  }
  throw ANN::Exception{
      CURRENT_FUNCTION,
      "Unknown leaky relu configuration provided '" + config +
          "'. Allowed configurations are: 'alpha'. From line " + lineNumStr};
}
void ModelLoader::configLayer(Softmax &layer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr) {
  throw ANN::Exception{CURRENT_FUNCTION,
                       "No softmax configuration supported. From line " +
                           lineNumStr};
}

void ModelLoader::configOptimizer(SGD &optimizer, const std::string &config,
                                  const std::string &value,
                                  const std::string &lineNumStr) {
  if (config == "learning_rate") {
    optimizer.learningRate = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "decay") {
    optimizer.decay = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "momentum") {
    optimizer.momentum = parseStrictFloat(value, lineNumStr);
    return;
  }
  throw ANN::Exception{CURRENT_FUNCTION,
                       "Unknown SGD configuration provided '" + config +
                           "'. Allowed configurations are: 'learning_rate', "
                           "'decay', 'momentum'. From line " +
                           lineNumStr};
}
void ModelLoader::configOptimizer(AdaGrad &optimizer, const std::string &config,
                                  const std::string &value,
                                  const std::string &lineNumStr) {
  if (config == "learning_rate") {
    optimizer.learningRate = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "decay") {
    optimizer.decay = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "epsilon") {
    optimizer.epsilon = parseStrictFloat(value, lineNumStr);
    return;
  }
  throw ANN::Exception{CURRENT_FUNCTION,
                       "Unknown AdaGrad configuration provided '" + config +
                           "'. Allowed configurations are: 'learning_rate', "
                           "'decay', 'epsilon'. From line " +
                           lineNumStr};
}
void ModelLoader::configOptimizer(RMSProp &optimizer, const std::string &config,
                                  const std::string &value,
                                  const std::string &lineNumStr) {
  if (config == "learning_rate") {
    optimizer.learningRate = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "decay") {
    optimizer.decay = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "epsilon") {
    optimizer.epsilon = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "rho") {
    optimizer.rho = parseStrictFloat(value, lineNumStr);
    return;
  }
  throw ANN::Exception{CURRENT_FUNCTION,
                       "Unknown RMSProp configuration provided '" + config +
                           "'. Allowed configurations are: 'learning_rate', "
                           "'decay', 'epsilon', 'rho'. From line " +
                           lineNumStr};
}
void ModelLoader::configOptimizer(Adam &optimizer, const std::string &config,
                                  const std::string &value,
                                  const std::string &lineNumStr) {
  if (config == "learning_rate") {
    optimizer.learningRate = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "decay") {
    optimizer.decay = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "epsilon") {
    optimizer.epsilon = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "beta1") {
    optimizer.beta1 = parseStrictFloat(value, lineNumStr);
    return;
  }
  if (config == "beta2") {
    optimizer.beta2 = parseStrictFloat(value, lineNumStr);
    return;
  }
  throw ANN::Exception{CURRENT_FUNCTION,
                       "Unknown Adam configuration provided '" + config +
                           "'. Allowed configurations are: 'learning_rate', "
                           "'decay', 'epsilon', 'beta1', 'beta2'. From line " +
                           lineNumStr};
}

void ModelLoader::trim(std::string &str) {
  // Comment trim - erases from first '#' character to str end
  str.erase(std::find_if(str.begin(), str.end(),
                         [](unsigned char c) { return c == '#'; }),
            str.end());

  // left trim - erases from str start to first non-space character
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(),
                         [](unsigned char c) { return !std::isspace(c); }));

  // right trim - erases from first non-space character (from end) to str end
  str.erase(std::find_if(str.rbegin(), str.rend(),
                         [](unsigned char c) { return !std::isspace(c); })
                .base(),
            str.end());
}

void ModelLoader::split(std::string_view str, char c, std::string &left,
                        std::string &right, const std::string &lineNumStr) {
  auto foundC{str.find(c)};
  if (foundC == std::string::npos)
    throw ANN::Exception{CURRENT_FUNCTION, "Expected '" + std::string{1, c} +
                                               "'. From line " + lineNumStr};

  left = str.substr(0, foundC);
  right = str.substr(foundC + 1);
  trim(left);
  trim(right);
}

int ModelLoader::parseStrictInt(const std::string &str,
                                const std::string &lineNumStr) {
  size_t pos{};
  int val{};

  try {
    val = std::stoi(str, &pos);
  } catch (...) {
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Expected an integer value. From line " + lineNumStr};
  };

  if (pos != str.size())
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Expected an integer value. From line " + lineNumStr};

  return val;
}

float ModelLoader::parseStrictFloat(const std::string &str,
                                    const std::string &lineNumStr) {
  size_t pos{};
  float val{};

  try {
    val = std::stof(str, &pos);
  } catch (...) {
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Expected a floating point value. From line " +
                             lineNumStr};
  };

  if (pos != str.size())
    throw ANN::Exception{CURRENT_FUNCTION,
                         "Expected a floating point value. From line " +
                             lineNumStr};

  return val;
}

bool ModelLoader::parseStrictBool(const std::string &str,
                                  const std::string &lineNumStr) {
  if (str == "true")
    return true;
  if (str == "false")
    return false;
  throw ANN::Exception{CURRENT_FUNCTION,
                       "Expected a boolean value. From line " + lineNumStr};
}
} // namespace ANN
