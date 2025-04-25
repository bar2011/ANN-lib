#pragma once

#include "math/matrix.h"
#include <array>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace MNist {
class Loader {
public:
  // a type which contains a vector of labels and a matching vector of images
  using DataPair = std::tuple<std::vector<unsigned char>,
                              std::shared_ptr<Math::Matrix<unsigned char>>>;

  Loader(std::string_view trainingLabelsPath,
         std::string_view trainingImagesPath,
         std::string_view testingLabelsPath, std::string_view testingImagesPath)
      : m_trainingLabelsPath{trainingLabelsPath},
        m_trainingImagesPath{trainingImagesPath},
        m_testingLabelsPath{testingLabelsPath},
        m_testingImagesPath{testingImagesPath} {}

  // Load all data (training and testing) from paths given
  // Returns in an array where the first item is training data, second is
  // testing. Each array item is a tuple, where the first element is a
  // std::vector of labels (each item in the vector is a label), and the second
  // element of the tuple is a Math::Matrix, where each row corresponds to the
  // data of each image, where each image has size 28x28, so the matrix has 28 *
  // 28 = 784 columns.
  std::array<DataPair, 2> loadData() const;

private:
  std::string m_trainingLabelsPath{};
  std::string m_trainingImagesPath{};
  std::string m_testingLabelsPath{};
  std::string m_testingImagesPath{};

  static DataPair loadImages(const std::string &labelsPath,
                             const std::string &imagesPath);

  static std::vector<unsigned char>
  loadLabelsFile(const std::string &labelsPath);

  static std::shared_ptr<Math::Matrix<unsigned char>>
  loadImagesFile(const std::string &imagesPath);

  static std::vector<unsigned char> readBytes(std::ifstream &file,
                                              const unsigned int length,
                                              const std::string &filename = "");

  static unsigned int readU32(std::ifstream &file,
                              const std::string &filename = "");
};
} // namespace MNist
