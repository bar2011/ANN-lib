#pragma once

#include "ann/feedForwardModel.h"
#include "ann/modelDescriptors.h"
#include <string>
#include <string_view>

namespace ANN {
class ModelLoader {
public:
  [[nodiscard]] static FeedForwardModel
  loadFeedForward(const std::string &path);

private:
  // Handles changing layer configuration by type (for easy use with std::visit)
  static void configLayer(Dense &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);
  static void configLayer(Dropout &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);
  static void configLayer(Step &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);
  static void configLayer(Sigmoid &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);
  static void configLayer(ReLU &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);
  static void configLayer(LeakyReLU &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);
  static void configLayer(Softmax &layer, const std::string &config,
                          const std::string &value,
                          const std::string &lineNumStr);

  // Handles changing optimizer configuration by type (for easy use with
  // std::visit)
  static void configOptimizer(SGD &optimizer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr);
  static void configOptimizer(AdaGrad &optimizer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr);
  static void configOptimizer(RMSProp &optimizer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr);
  static void configOptimizer(Adam &optimizer, const std::string &config,
                              const std::string &value,
                              const std::string &lineNumStr);

  // Trims whitespace from str (left and right)
  static void trim(std::string &str);

  // Splits given input along character and trims
  static void split(std::string_view str, char c, std::string &left,
                    std::string &right, const std::string &lineNumStr);

  // Parses an integer
  // Expects the whole string to be an integer, otherwise throws ANN::Exception
  // lineNum is for exception formatting
  static int parseStrictInt(const std::string &str,
                            const std::string &lineNumStr);

  // Parses a float
  // Expects the whole string to be a float, otherwise throws ANN::Exception
  // lineNum is for exception formatting
  static float parseStrictFloat(const std::string &str,
                                const std::string &lineNumStr);

  // Parses a boolearn
  // Expects the whole string to be equal 'true' or 'false', otherwise throw
  // ANN::Exception
  // lineNum is for exception formatting
  static bool parseStrictBool(const std::string &str,
                              const std::string &lineNumStr);
};
} // namespace ANN
