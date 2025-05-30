#include "ann/modelDescriptors.h"
#include "loaders/csv.h"
#include "loaders/mnist.h"

#include "ann/feedForwardModel.h"
#include "ann/modelDescriptors.h"

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
    trainMNist();
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

  ANN::FeedForwardModelDescriptor modelDesc{
      .inputs = inputs,
      .layers = {ANN::Dense{
                     .neurons = 64,
                     .initMethod = ANN::WeightInit::He,
                 },
                 ANN::LeakyReLU{.alpha = 1e-2}, ANN::Dropout{.dropRate = 0.1},
                 ANN::Dense{
                     .neurons = 64,
                     .initMethod = ANN::WeightInit::He,
                 },
                 ANN::LeakyReLU{.alpha = 1e-2}, ANN::Dropout{.dropRate = 0.1},
                 ANN::Dense{
                     .neurons = 32,
                     .initMethod = ANN::WeightInit::He,
                 },
                 ANN::LeakyReLU{.alpha = 1e-2},
                 ANN::Dense{.neurons = outputs,
                            .initMethod = ANN::WeightInit::Xavier}}};

  ANN::FeedForwardTrainingDescriptor trainDesc{
      .loss = ANN::MeanSquaredErrorLoss{},
      .optimizer = ANN::Adam{.learningRate = 2e-3f, .decay = 1e-3f},
      .batchSize = batchSize,
      .epochs = 5,
      .shuffleBatches = true,
      .verbose = true};

  auto model{std::make_unique<ANN::FeedForwardModel>(modelDesc, trainDesc)};

  model->train(*trainData.first, *trainData.second);

  auto [testData, testLabels]{loader->getTest()};
  // dense1->forward(std::move(testData));
  // activation1->forward(dense1->output());
  // dense2->forward(activation1->output());
  // activation2->forward(dense2->output());
  // dense3->forward(activation2->output());
  // activation3->forward(dense3->output());
  // outputLayer->forward(activation3->output());
  // outputLoss->forward(outputLayer->output(), std::move(testLabels));
  //
  // std::cout << "Test loss: " << outputLoss->mean() << '\n';
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

  ANN::FeedForwardModelDescriptor modelDesc{
      .inputs = inputs,
      .layers = {
          ANN::Dense{
              .neurons = 64,
              .initMethod = ANN::WeightInit::He,
          },
          ANN::LeakyReLU{.alpha = 1e-2}, ANN::Dropout{.dropRate = 0.1},
          ANN::Dense{
              .neurons = 64,
              .initMethod = ANN::WeightInit::He,
          },
          ANN::LeakyReLU{.alpha = 1e-2}, ANN::Dropout{.dropRate = 0.1},
          ANN::Dense{
              .neurons = 32,
              .initMethod = ANN::WeightInit::He,
          },
          ANN::LeakyReLU{.alpha = 1e-2},
          ANN::Dense{.neurons = outputs, .initMethod = ANN::WeightInit::Xavier},
          ANN::Sigmoid{}}};

  ANN::FeedForwardTrainingDescriptor trainDesc{
      .loss = ANN::BinaryCrossEntropyLoss{},
      .optimizer = ANN::Adam{.learningRate = 2e-3f, .decay = 1e-3f},
      .batchSize = batchSize,
      .epochs = 5,
      .shuffleBatches = true,
      .verbose = true};

  auto model{std::make_unique<ANN::FeedForwardModel>(modelDesc, trainDesc)};

  model->train(*trainData.first, *trainData.second);

  // auto [testData, testLabels]{loader->getTest()};
  // dense1->forward(std::move(testData));
  // activation1->forward(dense1->output());
  // dense2->forward(activation1->output());
  // activation2->forward(dense2->output());
  // dense3->forward(activation2->output());
  // activation3->forward(dense3->output());
  // outputLayer->forward(activation3->output());
  // outputActivation->forward(outputLayer->output());
  // outputLoss->forward(outputActivation->output(), std::move(testLabels));
  //
  // std::cout << "Test loss: " << outputLoss->mean()
  //           << "\nTest accuracy: " << outputLoss->accuracy() << '\n';
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

  ANN::FeedForwardModelDescriptor modelDesc{
      .inputs = 28 * 28,
      .layers = {
          ANN::Dense{.neurons = 32,
                     .initMethod = ANN::WeightInit::He,
                     .l2Weight = 1e-5f,
                     .l2Bias = 1e-5f},
          ANN::LeakyReLU{.alpha = 1e-2f},
          ANN::Dense{.neurons = 16,
                     .initMethod = ANN::WeightInit::He,
                     .l2Weight = 1e-5f,
                     .l2Bias = 1e-5f},
          ANN::LeakyReLU{.alpha = 1e-2f},
          ANN::Dense{.neurons = 10, .initMethod = ANN::WeightInit::He},
      }};

  ANN::FeedForwardTrainingDescriptor trainDesc{
      .loss = ANN::CategoricalCrossEntropySoftmaxLoss{},
      .optimizer = ANN::Adam{.learningRate = 2e-2f, .decay = 3e-4f},
      .batchSize = 64,
      .epochs = 5,
      .trainValidationRate = 0.05,
      .shuffleBatches = true,
      .verbose = true,
  };

  auto model{std::make_unique<ANN::FeedForwardModel>(modelDesc, trainDesc)};

  model->train(*trainingImages, *trainingLabels);

  // dense1->forward(testingImages);
  // activation1->forward(dense1->output());
  // dense2->forward(activation1->output());
  // activation2->forward(dense2->output());
  // outputLayer->forward(activation2->output());
  // outputSoftmaxLoss->forward(outputLayer->output(), testingLabels);
  //
  // std::cout << "Test loss: " << outputSoftmaxLoss->mean()
  //           << "\nTest accuracy: " << outputSoftmaxLoss->accuracy() << '\n';
}
