#pragma once

#include "math/matrix.h"
#include <array>
#include <fstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace MNist {
class Loader {
public:
  // a type which contains a vector of labels and a matching vector of images
  using DataPair =
      std::tuple<std::vector<double>, std::vector<Math::Matrix<double>>>;

  Loader(std::string_view trainingLabelsPath,
         std::string_view trainingImagesPath,
         std::string_view testingLabelsPath, std::string_view testingImagesPath)
      : m_trainingLabelsPath{trainingLabelsPath},
        m_trainingImagesPath{trainingImagesPath},
        m_testingLabelsPath{testingLabelsPath},
        m_testingImagesPath{testingImagesPath} {}

  // Load all data (training and testing) from paths given
  std::array<DataPair, 2> loadData() const;

private:
  std::string m_trainingLabelsPath{};
  std::string m_trainingImagesPath{};
  std::string m_testingLabelsPath{};
  std::string m_testingImagesPath{};

  static DataPair loadImages(std::string_view labelsPath,
                             std::string_view imagesPath);

  static std::vector<double> loadLabelsFile(std::string_view labelsPath);

  static std::vector<Math::Matrix<double>>
  loadImagesFile(std::string_view imagesPath);

  static std::vector<double> readBytes(std::ifstream &file,
                                       const unsigned int length,
                                       const std::string &filename = "");

  static unsigned int readU32(std::ifstream &file,
                              const std::string &filename = "");
};
} // namespace MNist
