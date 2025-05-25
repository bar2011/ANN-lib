#include "loaders/csv.h"

#include "loaders/exception.h"

namespace Loaders {
CSV::CSV(const std::string &trainDataPath, const std::string &trainLabelPath,
         const std::string &testDataPath, const std::string &testLabelPath,
         size_t batchSize, size_t testSize, size_t features, size_t targets)
    : m_trainDataFile{trainDataPath}, m_trainLabelsFile{trainLabelPath},
      m_testDataFile{testDataPath}, m_testLabelsFile{testLabelPath},
      m_batchSize{batchSize}, m_testSize{testSize}, m_features{features},
      m_targets{targets} {}

std::pair<std::unique_ptr<Math::Matrix<float>>,
          std::unique_ptr<Math::Matrix<float>>>
CSV::getTrainBatch(size_t batchNum) {
  for (size_t i{}; i < batchNum * m_batchSize; ++i) {
    if (!m_trainDataFile.ignore(std::numeric_limits<std::streamsize>::max(),
                                '\n'))
      throw Loaders::Exception{"Loaders::CSV::getTrainBatch(size_t)",
                               "Reached EOF before got to start of batch data"};

    if (!m_trainLabelsFile.ignore(std::numeric_limits<std::streamsize>::max(),
                                  '\n'))
      throw Loaders::Exception{
          "Loaders::CSV::getTrainBatch(size_t)",
          "Reached EOF before got to start of batch labels"};
  }

  auto batchData{std::make_unique<Math::Matrix<float>>(
      m_batchSize, m_features, [&trainDataFile = m_trainDataFile]() {
        float f{};
        if (!(trainDataFile >> f))
          throw Loaders::Exception{
              "Loaders::CSV::getTrainBatch(size_t)",
              "Reached EOF before got to end of batch data"};

        // Skip comma/newline after data if exists
        if (!trainDataFile.get())
          trainDataFile.clear();

        return f;
      })};

  auto batchLabels{std::make_unique<Math::Matrix<float>>(
      m_batchSize, m_targets, [&trainLabelsFile = m_trainLabelsFile]() {
        float f{};
        if (!(trainLabelsFile >> f))
          throw Loaders::Exception{
              "Loaders::CSV::getTrainBatch(size_t)",
              "Reached EOF before got to end of batch labels"};

        // Skip comma/newline after data if exists
        if (!trainLabelsFile.get())
          trainLabelsFile.clear();

        return f;
      })};

  // Put fstreams back on the file start
  m_trainDataFile.clear();
  m_trainDataFile.seekg(0);
  m_trainLabelsFile.clear();
  m_trainLabelsFile.seekg(0);

  return std::pair{std::move(batchData), std::move(batchLabels)};
}

std::pair<std::unique_ptr<Math::Matrix<float>>,
          std::unique_ptr<Math::Matrix<float>>>
CSV::getTest() {
  auto testData{std::make_unique<Math::Matrix<float>>(
      m_testSize, m_features, [&testDataFile = m_testDataFile]() {
        float f{};
        if (!(testDataFile >> f))
          throw Loaders::Exception{
              "Loaders::CSV::getTest()",
              "Reached EOF before got to end of test data"};

        // Skip comma/newline after data if exists
        if (!testDataFile.get())
          testDataFile.clear();

        return f;
      })};

  auto testLabels{std::make_unique<Math::Matrix<float>>(
      m_testSize, m_targets, [&testLabelsFile = m_testLabelsFile]() {
        float f{};
        if (!(testLabelsFile >> f))
          throw Loaders::Exception{
              "Loaders::CSV::getTest()",
              "Reached EOF before got to end of test labels"};

        // Skip comma/newline after data if exists
        if (!testLabelsFile.get())
          testLabelsFile.clear();

        return f;
      })};

  // Put fstreams back on the file start
  m_testDataFile.clear();
  m_testDataFile.seekg(0);
  m_testLabelsFile.clear();
  m_testLabelsFile.seekg(0);

  return std::pair{std::move(testData), std::move(testLabels)};
}
} // namespace Loaders
