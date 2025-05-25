#pragma once

#include "math/matrix.h"

#include <fstream>
#include <memory>
#include <string>
#include <utility>

namespace Loaders {
// Class which loads a CSV file into a model-accepted form
class CSV {
public:
  // batchSize - size of a single batch
  // testSize - number of testing samples
  // features - number of inputs to the network
  // targets - number of outputs of the network
  CSV(const std::string &trainDataPath, const std::string &trainLabelPath,
      const std::string &testDataPath, const std::string &testLabelPath,
      size_t batchSize, size_t testSize, size_t features, size_t targets);

  // Returns a pair of <data, labels> of the given batch
  // Throws if reached EOF before got to corresponding batch
  std::pair<std::unique_ptr<Math::Matrix<float>>,
            std::unique_ptr<Math::Matrix<float>>>
  getTrainBatch(size_t batchNum);

  // Returns a pair of <data, labels> of all test data
  std::pair<std::unique_ptr<Math::Matrix<float>>,
            std::unique_ptr<Math::Matrix<float>>>
  getTest();

private:
  std::ifstream m_trainDataFile{};
  std::ifstream m_trainLabelsFile{};
  std::ifstream m_testDataFile{};
  std::ifstream m_testLabelsFile{};

  size_t m_batchSize{};
  size_t m_testSize{};
  size_t m_features{};
  size_t m_targets{};
};
} // namespace Loaders
