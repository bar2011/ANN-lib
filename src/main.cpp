#include "ann/modelLoader.h"
#include "loaders/csv.h"
#include "loaders/mnist.h"

#include "math/matrixBase.h"

#include <iostream>
#include <stdexcept>

template <typename T> void printMatrixImage(const Math::MatrixBase<T> &m) {
  for (size_t row{}; row < m.rows(); ++row) {
    for (size_t col{}; col < m.cols(); ++col)
      std::cout << "\033[38;2;" << static_cast<unsigned int>(m[row, col] * 255)
                << ';' << static_cast<unsigned int>(m[row, col] * 255) << ';'
                << static_cast<unsigned int>(m[row, col] * 255) << "m█\033[0m";
    std::cout << '\n';
  }
}

void trainRegression();
void trainBinaryLogisticRegression();
void trainMNist();

int main() {
  try {
    // 0 - mnist
    // 1 - binary
    // 2 - regression
    int mode{};
    std::cout
        << "Which model to run? (0 - mnist, 1 - binary, 2 - regression)\n";
    std::cin >> mode;
    switch (mode) {
    case 0:
      trainMNist();
      break;
    case 1:
      trainBinaryLogisticRegression();
      break;
    case 2:
      trainRegression();
      break;
    default:
      std::cout << "I expected better of you.\n";
    }
  } catch (std::runtime_error &e) {
    std::cout << "An error occured: " << e.what() << '\n';
  } catch (...) {
    std::cout << "An unknown error occured\n";
  }

  return 0;
}

void trainRegression() {
  constexpr unsigned short inputs{20};
  constexpr unsigned short outputs{3};
  constexpr unsigned int batchSize{64};
  constexpr unsigned int trainingSize{50'000};
  constexpr unsigned int testingSize{10'000};

  auto loader{std::make_unique<Loaders::CSV>(
      "data/regression-train-data.csv", "data/regression-train-labels.csv",
      "data/regression-test-data.csv", "data/regression-test-labels.csv",
      batchSize, trainingSize, testingSize, inputs, outputs)};

  auto trainData{loader->getTrainData()};

  auto model{ANN::ModelLoader::loadFeedForward("regression.model")};

  model.train(trainData.first, trainData.second);

  auto [testData, testLabels]{loader->getTest()};

  model.evaluate(testData, testLabels);
  float dataLoss{};
  model.calculateLoss(&dataLoss);
  float accuracy{model.calculateAccuracy()};
  std::cout << "\nTest loss: " << dataLoss << '\n';
  if (accuracy != -1)
    std::cout << "Test accuracy: " << accuracy << "\n\n";
}

void trainBinaryLogisticRegression() {
  constexpr unsigned short inputs{50};
  constexpr unsigned short outputs{7};
  constexpr unsigned int batchSize{64};
  constexpr unsigned int trainingSize{50'000};
  constexpr unsigned int testingSize{10'000};

  auto loader{std::make_unique<Loaders::CSV>(
      "data/multilabel-train-data.csv", "data/multilabel-train-labels.csv",
      "data/multilabel-test-data.csv", "data/multilabel-test-labels.csv",
      batchSize, trainingSize, testingSize, inputs, outputs)};

  auto trainData{loader->getTrainData()};

  auto model{ANN::ModelLoader::loadFeedForward("binary.model")};

  model.train(trainData.first, trainData.second);

  auto [testData, testLabels]{loader->getTest()};

  model.evaluate(testData, testLabels);
  float dataLoss{};
  model.calculateLoss(&dataLoss);
  float accuracy{model.calculateAccuracy()};
  std::cout << "\nTest loss: " << dataLoss << '\n';
  if (accuracy != -1)
    std::cout << "Test accuracy: " << accuracy << "\n\n";
}

void trainMNist() {
  auto loader{std::make_unique<Loaders::MNist>(
      "data/train-labels-idx1-ubyte", "data/train-images-idx3-ubyte",
      "data/t10k-labels-idx1-ubyte", "data/t10k-images-idx3-ubyte")};
  std::array<Loaders::MNist::DataPair, 2> data{loader->loadData()};

  auto trainingImages{std::get<1>(data[0])};
  auto trainingLabels{std::get<0>(data[0])};
  auto testingImages{std::get<1>(data[1])};
  auto testingLabels{std::get<0>(data[1])};

  auto model{ANN::ModelLoader::loadFeedForward("mnist.model")};
  std::cout << "Loaded model successfully.\n";

  // model->loadParams("mnist.data");
  //
  // std::cout << "Loaded parameters successfully.\n";

  model.train(trainingImages, trainingLabels, "mnist.log");

  model.evaluate(testingImages, testingLabels);
  float dataLoss{};
  model.calculateLoss(&dataLoss);
  float accuracy{model.calculateAccuracy()};
  std::cout << "\nTest loss: " << dataLoss << '\n';
  if (accuracy != -1)
    std::cout << "Test accuracy: " << accuracy << "\n\n";

  std::cout << "Trained successfully.\n";

  model.saveParams("mnist.data");

  std::cout << "Saved successfully.\n";
}
