#include "loaders/mnist.h"

#include "loaders/exception.h"
#include "utils/exceptions.h"

#include <array>
#include <fstream>
#include <string>
#include <tuple>

namespace Loaders {
std::array<MNist::DataPair, 2> MNist::loadData() const {
  auto train{loadImages(m_trainingLabelsPath, m_trainingImagesPath)};
  auto test{loadImages(m_testingLabelsPath, m_testingImagesPath)};

  return std::array{std::move(train), std::move(test)};
}

MNist::DataPair MNist::loadImages(const std::string &labelsPath,
                                  const std::string &imagesPath) {
  auto labels{loadLabelsFile(labelsPath)};
  auto images{loadImagesFile(imagesPath)};

  return std::make_tuple(std::move(labels), std::move(images));
}

std::shared_ptr<Math::Vector<float>>
MNist::loadLabelsFile(const std::string &labelsPath) {
  std::ifstream labelsFile{labelsPath, std::ios::binary | std::ios::in};

  // testing number which should always be 2049
  unsigned int magicNumber{readU32(labelsFile, labelsPath)};
  if (magicNumber != 2049)
    throw Loaders::Exception{
        CURRENT_FUNCTION,
        "Invalid file format for image label file: " + labelsPath +
            "\n Magic number mismatch (expected 2049)"};

  // Number of image labels (each one is a byte, so also length of remaining
  // file)
  unsigned int size{readU32(labelsFile, labelsPath)};

  auto labels{std::make_shared<Math::Vector<float>>(size)};

  labels->fill(
      [&labelsFile, &labelsPath](float *item) {
        unsigned char byte{};
        if (!labelsFile.read(reinterpret_cast<char *>(&byte), 1))
          throw Loaders::Exception{
              CURRENT_FUNCTION,
              "Can't read file " + labelsPath +
                  ": file size smaller then needed to read all images."};
        *item = static_cast<float>(byte);
      },
      false);

  return labels;
}

std::shared_ptr<Math::Matrix<float>>
MNist::loadImagesFile(const std::string &imagesPath) {
  std::ifstream imagesFile{imagesPath, std::ios::binary | std::ios::in};

  // testing number which should always be 2051
  unsigned int magicNumber{readU32(imagesFile, imagesPath)};
  if (magicNumber != 2051)
    throw Loaders::Exception{
        CURRENT_FUNCTION, "Invalid file format for images file: " + imagesPath +
                              "\n Magic number mismatch (expected 2051)"};

  // Number of images
  unsigned int size{readU32(imagesFile, imagesPath)};

  // Number of rows in each image
  unsigned int rows{readU32(imagesFile, imagesPath)};

  // Number of columns in each image
  unsigned int cols{readU32(imagesFile, imagesPath)};

  auto images{std::make_unique<Math::Matrix<float>>(size, rows * cols)};

  images->fill(
      [&imagesFile, &imagesPath](float *item) {
        unsigned char byte{};
        if (!imagesFile.read(reinterpret_cast<char *>(&byte), 1))
          throw Loaders::Exception{
              CURRENT_FUNCTION,
              "Can't read file " + imagesPath +
                  ": file size smaller then needed to read all images."};
        *item = static_cast<float>(byte) / 255.0f;
      },
      false);

  return images;
}

unsigned int MNist::readU32(std::ifstream &file, const std::string &filename) {
  std::array<char, 4> bytes{};
  if (!file.read(bytes.begin(), 4))
    throw Loaders::Exception{CURRENT_FUNCTION,
                             "Can't read file " + filename + ": reached end"};
  unsigned int value{
      static_cast<unsigned int>(static_cast<unsigned char>(bytes[3]) |
                                (static_cast<unsigned char>(bytes[2]) << 8) |
                                (static_cast<unsigned char>(bytes[1]) << 16) |
                                (static_cast<unsigned char>(bytes[0]) << 24))};

  return value;
}
} // namespace Loaders
