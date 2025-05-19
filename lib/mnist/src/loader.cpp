#include "mnist/loader.h"

#include "mnist/exception.h"

#include <array>
#include <fstream>
#include <string>
#include <tuple>

std::array<MNist::Loader::DataPair, 2> MNist::Loader::loadData() const {
  auto train{loadImages(m_trainingLabelsPath, m_trainingImagesPath)};
  auto test{loadImages(m_testingLabelsPath, m_testingImagesPath)};

  return std::array{std::move(train), std::move(test)};
}

MNist::Loader::DataPair
MNist::Loader::loadImages(const std::string &labelsPath,
                          const std::string &imagesPath) {
  std::shared_ptr<Math::Vector<unsigned short>> labels{
      loadLabelsFile(labelsPath)};
  std::shared_ptr<Math::Matrix<float>> images{loadImagesFile(imagesPath)};

  return std::make_tuple(std::move(labels), std::move(images));
}

std::shared_ptr<Math::Vector<unsigned short>>
MNist::Loader::loadLabelsFile(const std::string &labelsPath) {
  std::ifstream labelsFile{labelsPath, std::ios::binary | std::ios::in};

  // testing number which should always be 2049
  unsigned int magicNumber{readU32(labelsFile, labelsPath)};
  if (magicNumber != 2049)
    throw MNist::Exception{
        "MNist::Loader::loadLabelsFile(const std::string&)",
        "Invalid file format for image label file: " + labelsPath +
            "\n Magic number mismatch (expected 2049)"};

  // Number of image labels (each one is a byte, so also length of remaining
  // file)
  unsigned int size{readU32(labelsFile, labelsPath)};

  auto labels{std::make_shared<Math::Vector<unsigned short>>(size)};

  labels->fill(
      [&labelsFile, &labelsPath](unsigned short *item) {
        unsigned char byte{};
        if (!labelsFile.read(reinterpret_cast<char *>(&byte), 1))
          throw MNist::Exception{
              "MNist::Loader::loadImagesFile(const std::string&)",
              "Can't read file " + labelsPath +
                  ": file size smaller then needed to read all images."};
        *item = static_cast<unsigned short>(byte);
      },
      false);

  return labels;
}

std::shared_ptr<Math::Matrix<float>>
MNist::Loader::loadImagesFile(const std::string &imagesPath) {
  std::ifstream imagesFile{imagesPath, std::ios::binary | std::ios::in};

  // testing number which should always be 2051
  unsigned int magicNumber{readU32(imagesFile, imagesPath)};
  if (magicNumber != 2051)
    throw MNist::Exception{
        "MNist::Loader::loadImagesFile(const std::string&)",
        "Invalid file format for images file: " + imagesPath +
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
          throw MNist::Exception{
              "MNist::Loader::loadImagesFile(const std::string&)",
              "Can't read file " + imagesPath +
                  ": file size smaller then needed to read all images."};
        *item = static_cast<float>(byte) / 255.0f;
      },
      false);

  return images;
}

unsigned int MNist::Loader::readU32(std::ifstream &file,
                                    const std::string &filename) {
  std::array<char, 4> bytes{};
  if (!file.read(bytes.begin(), 4))
    throw MNist::Exception{
        "MNist::Loader::readU32(std::ifstream&, const std::string&)",
        "Can't read file " + filename + ": reached end"};
  unsigned int value{
      static_cast<unsigned int>(static_cast<unsigned char>(bytes[3]) |
                                (static_cast<unsigned char>(bytes[2]) << 8) |
                                (static_cast<unsigned char>(bytes[1]) << 16) |
                                (static_cast<unsigned char>(bytes[0]) << 24))};

  return value;
}
