#include "loaders/csv.h"

#include "loaders/exception.h"
#include "utils/exceptions.h"

#include <sstream>

namespace Loaders {
CSV::CSV(const std::string &trainDataPath, const std::string &trainLabelPath,
         const std::string &testDataPath, const std::string &testLabelPath,
         size_t batchSize, size_t trainSize, size_t testSize, size_t features,
         size_t targets)
    : m_trainDataFile{trainDataPath}, m_trainLabelsFile{trainLabelPath},
      m_testDataFile{testDataPath}, m_testLabelsFile{testLabelPath},
      m_batchSize{batchSize}, m_trainSize{trainSize}, m_testSize{testSize},
      m_features{features}, m_targets{targets} {}

std::pair<Math::Matrix<float>, Math::Matrix<float>>
CSV::getTrainBatch(size_t batchNum) {
  std::string line{};

  for (size_t i{}; i < batchNum * m_batchSize; ++i)
    if (!std::getline(m_trainDataFile, line) ||
        !std::getline(m_trainLabelsFile, line))
      throw Loaders::Exception{CURRENT_FUNCTION,
                               "Reached EOF before got to start of batch"};

  std::vector<float> dataVector{};
  std::vector<float> labelsVector{};
  dataVector.reserve(m_batchSize * m_features);
  dataVector.reserve(m_batchSize * m_targets);

  std::string token{};

  for (size_t batch{}; batch < m_batchSize; ++batch) {
    if (!std::getline(m_trainDataFile, line))
      throw Loaders::Exception{CURRENT_FUNCTION,
                               "Reached EOF while reading batch data"};

    std::istringstream ssData{line};
    for (size_t i{}; i < m_features; ++i) {
      if (!std::getline(ssData, token, ','))
        throw Loaders::Exception{CURRENT_FUNCTION, "Data field missing"};
      dataVector.push_back(std::stof(token));
    }

    if (!std::getline(m_trainLabelsFile, line))
      throw Loaders::Exception{CURRENT_FUNCTION,
                               "Reached EOF while reading batch labels"};

    std::istringstream ssLabels{line};
    for (size_t i{}; i < m_targets; ++i) {
      if (!std::getline(ssLabels, token, ','))
        throw Loaders::Exception{CURRENT_FUNCTION, "Label field missing"};
      labelsVector.push_back(std::stof(token));
    }
  }

  Math::Matrix<float> batchData{m_batchSize, m_features, std::move(dataVector)};

  Math::Matrix<float> batchLabels{m_batchSize, m_targets,
                                  std::move(labelsVector)};

  // Put fstreams back on the file start
  m_trainDataFile.clear();
  m_trainDataFile.seekg(0);
  m_trainLabelsFile.clear();
  m_trainLabelsFile.seekg(0);

  return std::pair{std::move(batchData), std::move(batchLabels)};
}

std::pair<Math::Matrix<float>, Math::Matrix<float>> CSV::getTrainData() {
  std::string line{};

  std::vector<float> dataVector{};
  std::vector<float> labelsVector{};
  dataVector.reserve(m_trainSize * m_features);
  labelsVector.reserve(m_trainSize * m_targets);

  std::string token{};

  for (size_t batch{}; batch < m_trainSize; ++batch) {
    if (!std::getline(m_trainDataFile, line))
      throw Loaders::Exception{CURRENT_FUNCTION,
                               "Reached EOF while reading data"};

    std::istringstream ssData{line};
    for (size_t i{}; i < m_features; ++i) {
      if (!std::getline(ssData, token, ','))
        throw Loaders::Exception{CURRENT_FUNCTION, "Data field missing"};
      dataVector.push_back(std::stof(token));
    }

    if (!std::getline(m_trainLabelsFile, line))
      throw Loaders::Exception{CURRENT_FUNCTION,
                               "Reached EOF while reading labels"};

    std::istringstream ssLabels{line};
    for (size_t i{}; i < m_targets; ++i) {
      if (!std::getline(ssLabels, token, ','))
        throw Loaders::Exception{CURRENT_FUNCTION, "Label field missing"};
      labelsVector.push_back(std::stof(token));
    }
  }

  Math::Matrix<float> trainData{m_trainSize, m_features, std::move(dataVector)};

  Math::Matrix<float> trainLabels{m_trainSize, m_targets,
                                  std::move(labelsVector)};

  // Put fstreams back on the file start
  m_trainDataFile.clear();
  m_trainDataFile.seekg(0);
  m_trainLabelsFile.clear();
  m_trainLabelsFile.seekg(0);

  return std::pair{std::move(trainData), std::move(trainLabels)};
}

std::pair<Math::Matrix<float>, Math::Matrix<float>> CSV::getTest() {
  Math::Matrix<float> testData{
      m_testSize, m_features, [&testDataFile = m_testDataFile]() {
        float f{};
        if (!(testDataFile >> f))
          throw Loaders::Exception{
              CURRENT_FUNCTION, "Reached EOF before got to end of test data"};

        // Skip comma/newline after data if exists
        if (!testDataFile.get())
          testDataFile.clear();

        return f;
      }};

  Math::Matrix<float> testLabels{
      m_testSize, m_targets, [&testLabelsFile = m_testLabelsFile]() {
        float f{};
        if (!(testLabelsFile >> f))
          throw Loaders::Exception{
              CURRENT_FUNCTION, "Reached EOF before got to end of test labels"};

        // Skip comma/newline after data if exists
        if (!testLabelsFile.get())
          testLabelsFile.clear();

        return f;
      }};

  // Put fstreams back on the file start
  m_testDataFile.clear();
  m_testDataFile.seekg(0);
  m_testLabelsFile.clear();
  m_testLabelsFile.seekg(0);

  return std::pair{std::move(testData), std::move(testLabels)};
}
} // namespace Loaders
