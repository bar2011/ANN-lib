#include "loaders/csv.h"
#include "loaders/mnist.h"

#include "math/matrixBase.h"
#include "math/random.h"

#include "ann/layers/MSELoss.h"
#include "ann/layers/binaryLoss.h"
#include "ann/layers/categoricalLossSoftmax.h"
#include "ann/layers/dense.h"
#include "ann/layers/dropout.h"

#include "ann/activations/leakyRelu.h"
#include "ann/activations/sigmoid.h"

#include "ann/optimizers/adam.h"

#include <iostream>
#include <stdexcept>

#include <chrono>

class Timer {
private:
  // Type aliases to make accessing nested type easier
  using Clock = std::chrono::high_resolution_clock;
  using Second = std::chrono::duration<double, std::ratio<1>>;

  std::chrono::time_point<Clock> m_beg{Clock::now()};

public:
  void reset() { m_beg = Clock::now(); }

  double elapsed() const {
    return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
  }
};

template <typename T> void printMatrixImage(const Math::MatrixBase<T> &m) {
  for (size_t row{}; row < m.rows(); ++row) {
    for (size_t col{}; col < m.cols(); ++col)
      std::cout << "\033[38;2;" << static_cast<unsigned int>(m[row, col] * 255)
                << ';' << static_cast<unsigned int>(m[row, col] * 255) << ';'
                << static_cast<unsigned int>(m[row, col] * 255) << "m█\033[0m";
    std::cout << '\n';
  }
}

std::string formatTime(double seconds) {
  if (seconds >= 1.0)
    return std::to_string(static_cast<unsigned long long>(seconds)) + "s";

  const unsigned long long ns = static_cast<unsigned long long>(seconds * 1e9);
  if (ns < 1000)
    return std::to_string(ns) + "ns";

  const unsigned long long us = ns / 1000;
  if (us < 1000)
    return std::to_string(us) + "us";

  const unsigned long long ms = us / 1000;
  if (ms < 1000)
    return std::to_string(ms) + "ms";

  // Fallback: round down to 0s
  return "0s";
}

void trainRegression();
void trainBinaryLogisticRegression();
void trainMNist();

int main() {
  try {
    trainRegression();
  } catch (std::runtime_error &e) {
    std::cout << "An error occured: " << e.what() << '\n';
  } catch (...) {
    std::cout << "An unknown error occured\n";
  }

  return 0;
}

void trainRegression() {
  constexpr unsigned short inputs{20};
  constexpr unsigned short layer1Neurons{64};
  constexpr unsigned short layer2Neurons{64};
  constexpr unsigned short layer3Neurons{32};
  constexpr unsigned short outputNeurons{3};
  constexpr unsigned int batchSize{64};
  constexpr unsigned int trainingSize{50'000};
  constexpr unsigned int testingSize{10'000};
  constexpr unsigned int epochs{5};
  constexpr float learningRate{2e-3};
  constexpr float learningRateDecay{1e-3};

  // normalized with:
  // data: (X + 4.959753276971317) / 9.772888927315826
  // labels: (Y + 857.3455055127644) / 1725.0832354415797
  auto loader{std::make_unique<Loaders::CSV>(
      "data/regression-train-data.csv", "data/regression-train-labels.csv",
      "data/regression-test-data.csv", "data/regression-test-labels.csv",
      batchSize, trainingSize, testingSize, inputs, outputNeurons)};

  auto trainData{loader->getTrainData()};

  auto dense1{std::make_unique<Layer::Dense>(inputs, layer1Neurons,
                                             Layer::WeightInit::He)};

  auto activation1{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto dropout1{std::make_unique<Layer::Dropout>(0.1)};

  auto dense2{std::make_unique<Layer::Dense>(layer1Neurons, layer2Neurons,
                                             Layer::WeightInit::He)};

  auto activation2{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto dropout2{std::make_unique<Layer::Dropout>(0.1)};

  auto dense3{std::make_unique<Layer::Dense>(layer2Neurons, layer3Neurons,
                                             Layer::WeightInit::He)};

  auto activation3{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto outputLayer{std::make_unique<Layer::Dense>(layer3Neurons, outputNeurons,
                                                  Layer::WeightInit::Xavier)};

  auto outputLoss{std::make_unique<Layer::MSELoss>()};

  auto optimizer{
      std::make_unique<Optimizers::Adam>(learningRate, learningRateDecay)};

  Timer trainingTimer{};
  Timer displayTimer{};

  constexpr size_t batches{trainingSize / batchSize};

  // Set up floating point printing for training updates
  std::cout << std::fixed << std::setprecision(4);

  for (size_t epoch{}; epoch < epochs; ++epoch) {

    std::cout << "Epoch " << epoch << ":\n";

    // Set up batch sequence
    std::vector<size_t> batchSequence(trainingSize / batchSize);
    std::iota(batchSequence.begin(), batchSequence.end(), 0);
    std::shuffle(batchSequence.begin(), batchSequence.end(), Math::Random::mt);

    trainingTimer.reset();
    for (size_t batch{}; batch < trainingSize / batchSize; ++batch) {

      // FORWARD PASS

      // Get batchSize training images and labels
      auto currentBatch{batchSequence[batch]};
      auto inputData{trainData.first->view(currentBatch * batchSize,
                                           (currentBatch + 1) * batchSize)};
      auto inputLabels{trainData.second->view(currentBatch * batchSize,
                                              (currentBatch + 1) * batchSize)};

      dense1->forward(std::move(inputData));
      activation1->forward(dense1->output());
      dropout1->forward(activation1->output());
      dense2->forward(dropout1->output());
      activation2->forward(dense2->output());
      dropout2->forward(activation2->output());
      dense3->forward(dropout2->output());
      activation3->forward(dense3->output());
      outputLayer->forward(activation3->output());
      outputLoss->forward(outputLayer->output(), std::move(inputLabels));

      float dataLoss{outputLoss->mean()};
      float regularizationLoss{outputLoss->regularizationLoss(*dense1) +
                               outputLoss->regularizationLoss(*dense2) +
                               outputLoss->regularizationLoss(*dense3)};
      float loss{dataLoss + regularizationLoss};

      // BACKWARD PASS
      outputLoss->backward();
      outputLayer->backward(outputLoss->dinputs());
      activation3->backward(outputLayer->dinputs());
      dense3->backward(activation3->dinputs());
      dropout2->backward(dense3->dinputs());
      activation2->backward(dropout2->dinputs());
      dense2->backward(activation2->dinputs());
      dropout1->backward(dense2->dinputs());
      activation1->backward(dropout1->dinputs());
      dense1->backward(activation1->dinputs());

      optimizer->preUpdate();
      optimizer->updateParams(*outputLayer);
      optimizer->updateParams(*dense3);
      optimizer->updateParams(*dense2);
      optimizer->updateParams(*dense1);
      optimizer->postUpdate();

      // Display information every about half second, or at the first/final
      // batch
      if (displayTimer.elapsed() >= 0.5 ||
          batch + 1 == trainingSize / batchSize || batch == 0) {
        std::cout << '\r' << batch + 1 << '/' << trainingSize / batchSize
                  << '\t' << static_cast<size_t>(trainingTimer.elapsed())
                  << "s " << formatTime(trainingTimer.elapsed() / (batch + 1))
                  << "/step \tloss: " << loss << " (data loss: " << dataLoss
                  << ", reg loss: " << regularizationLoss
                  << ") - lr: " << optimizer->learningRate()
                  << "                 " << std::flush;

        displayTimer.reset();
      }
    }

    std::cout << '\n';
  }

  auto [testData, testLabels]{loader->getTest()};
  dense1->forward(std::move(testData));
  activation1->forward(dense1->output());
  dense2->forward(activation1->output());
  activation2->forward(dense2->output());
  dense3->forward(activation2->output());
  activation3->forward(dense3->output());
  outputLayer->forward(activation3->output());
  outputLoss->forward(outputLayer->output(), std::move(testLabels));

  std::cout << "Test loss: " << outputLoss->mean() << '\n';
}

void trainBinaryLogisticRegression() {
  constexpr unsigned short inputs{50};
  constexpr unsigned short layer1Neurons{64};
  constexpr unsigned short layer2Neurons{64};
  constexpr unsigned short layer3Neurons{32};
  constexpr unsigned short outputNeurons{7};
  constexpr unsigned int batchSize{64};
  constexpr unsigned int trainingSize{50'000};
  constexpr unsigned int testingSize{10'000};
  constexpr unsigned int epochs{5};
  constexpr float learningRate{2e-3};
  constexpr float learningRateDecay{1e-3};

  auto loader{std::make_unique<Loaders::CSV>(
      "data/multilabel-train-data.csv", "data/multilabel-train-labels.csv",
      "data/multilabel-test-data.csv", "data/multilabel-test-labels.csv",
      batchSize, trainingSize, testingSize, inputs, outputNeurons)};

  auto trainData{loader->getTrainData()};

  auto dense1{std::make_unique<Layer::Dense>(inputs, layer1Neurons,
                                             Layer::WeightInit::He)};

  auto activation1{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto dropout1{std::make_unique<Layer::Dropout>(0.1)};

  auto dense2{std::make_unique<Layer::Dense>(layer1Neurons, layer2Neurons,
                                             Layer::WeightInit::He)};

  auto activation2{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto dropout2{std::make_unique<Layer::Dropout>(0.1)};

  auto dense3{std::make_unique<Layer::Dense>(layer2Neurons, layer3Neurons,
                                             Layer::WeightInit::He)};

  auto activation3{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto outputLayer{std::make_unique<Layer::Dense>(layer3Neurons, outputNeurons,
                                                  Layer::WeightInit::Xavier)};

  auto outputActivation{std::make_unique<Activation::Sigmoid>()};

  auto outputLoss{std::make_unique<Layer::BinaryLoss>()};

  auto optimizer{
      std::make_unique<Optimizers::Adam>(learningRate, learningRateDecay)};

  Timer trainingTimer{};
  Timer displayTimer{};

  constexpr size_t batches{trainingSize / batchSize};

  // Set up floating point printing for training updates
  std::cout << std::fixed << std::setprecision(4);

  for (size_t epoch{}; epoch < epochs; ++epoch) {

    std::cout << "Epoch " << epoch << ":\n";

    // Set up batch sequence
    std::vector<size_t> batchSequence(trainingSize / batchSize);
    std::iota(batchSequence.begin(), batchSequence.end(), 0);
    std::shuffle(batchSequence.begin(), batchSequence.end(), Math::Random::mt);

    trainingTimer.reset();
    for (size_t batch{}; batch < trainingSize / batchSize; ++batch) {

      // FORWARD PASS

      // Get batchSize training images and labels
      auto currentBatch{batchSequence[batch]};
      auto inputData{trainData.first->view(currentBatch * batchSize,
                                           (currentBatch + 1) * batchSize)};
      auto inputLabels{trainData.second->view(currentBatch * batchSize,
                                              (currentBatch + 1) * batchSize)};

      dense1->forward(std::move(inputData));
      activation1->forward(dense1->output());
      dropout1->forward(activation1->output());
      dense2->forward(dropout1->output());
      activation2->forward(dense2->output());
      dropout2->forward(activation2->output());
      dense3->forward(dropout2->output());
      activation3->forward(dense3->output());
      outputLayer->forward(activation3->output());
      outputActivation->forward(outputLayer->output());
      outputLoss->forward(outputActivation->output(), std::move(inputLabels));

      float dataLoss{outputLoss->mean()};
      float regularizationLoss{outputLoss->regularizationLoss(*dense1) +
                               outputLoss->regularizationLoss(*dense2) +
                               outputLoss->regularizationLoss(*dense3)};
      float loss{dataLoss + regularizationLoss};

      // BACKWARD PASS
      outputLoss->backward();
      outputActivation->backward(outputLoss->dinputs());
      outputLayer->backward(outputActivation->dinputs());
      activation3->backward(outputLayer->dinputs());
      dense3->backward(activation3->dinputs());
      dropout2->backward(dense3->dinputs());
      activation2->backward(dropout2->dinputs());
      dense2->backward(activation2->dinputs());
      dropout1->backward(dense2->dinputs());
      activation1->backward(dropout1->dinputs());
      dense1->backward(activation1->dinputs());

      optimizer->preUpdate();
      optimizer->updateParams(*outputLayer);
      optimizer->updateParams(*dense3);
      optimizer->updateParams(*dense2);
      optimizer->updateParams(*dense1);
      optimizer->postUpdate();

      // Display information every about half second, or at the first/final
      // batch
      if (displayTimer.elapsed() >= 0.5 ||
          batch + 1 == trainingSize / batchSize || batch == 0) {
        std::cout << '\r' << batch + 1 << '/' << trainingSize / batchSize
                  << '\t' << static_cast<size_t>(trainingTimer.elapsed())
                  << "s " << formatTime(trainingTimer.elapsed() / (batch + 1))
                  << "/step \taccuracy: " << outputLoss->accuracy()
                  << " - loss: " << loss << " (data loss: " << dataLoss
                  << ", reg loss: " << regularizationLoss
                  << ") - lr: " << optimizer->learningRate()
                  << "                 " << std::flush;

        displayTimer.reset();
      }
    }

    std::cout << '\n';
  }

  auto [testData, testLabels]{loader->getTest()};
  dense1->forward(std::move(testData));
  activation1->forward(dense1->output());
  dense2->forward(activation1->output());
  activation2->forward(dense2->output());
  dense3->forward(activation2->output());
  activation3->forward(dense3->output());
  outputLayer->forward(activation3->output());
  outputActivation->forward(outputLayer->output());
  outputLoss->forward(outputActivation->output(), std::move(testLabels));

  std::cout << "Test loss: " << outputLoss->mean()
            << "\nTest accuracy: " << outputLoss->accuracy() << '\n';
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

  constexpr unsigned int imageRows{28};
  constexpr unsigned int imageCols{28};
  constexpr unsigned short layer1Neurons{32};
  constexpr unsigned short layer2Neurons{16};
  constexpr unsigned short outputNeurons{10};
  constexpr unsigned int batchSize{64};
  constexpr unsigned int trainingSize{60'000};
  constexpr unsigned int epochs{1};
  constexpr float learningRate{2e-2};
  constexpr float learningRateDecay{3e-4};
  constexpr float learningRateMomentum{0.3};

  auto dense1{std::make_unique<Layer::Dense>(
      imageRows * imageCols, layer1Neurons, Layer::WeightInit::He, 0, 0, 1e-5f,
      1e-5f)};

  auto activation1{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto dropout1{std::make_unique<Layer::Dropout>(1e-1)};

  auto dense2{std::make_unique<Layer::Dense>(
      layer1Neurons, layer2Neurons, Layer::WeightInit::He, 0, 0, 1e-5f, 1e-5f)};

  auto activation2{std::make_unique<Activation::LeakyReLU>(1e-2)};

  auto dropout2{std::make_unique<Layer::Dropout>(5e-2)};

  auto outputLayer{std::make_unique<Layer::Dense>(layer2Neurons, outputNeurons,
                                                  Layer::WeightInit::He)};

  auto outputSoftmaxLoss{std::make_unique<Layer::CategoricalLossSoftmax>()};

  auto optimizer{
      std::make_unique<Optimizers::Adam>(learningRate, learningRateDecay)};

  Timer trainingTimer{};
  Timer displayTimer{};

  constexpr size_t batches{trainingSize / batchSize};

  // Set up floating point printing for training updates
  std::cout << std::fixed << std::setprecision(4);

  for (size_t epoch{}; epoch < epochs; ++epoch) {

    // Set up batch sequence
    std::vector<size_t> batchSequence(trainingSize / batchSize);
    std::iota(batchSequence.begin(), batchSequence.end(), 0);
    std::shuffle(batchSequence.begin(), batchSequence.end(), Math::Random::mt);

    trainingTimer.reset();
    for (size_t batch{}; batch < trainingSize / batchSize; ++batch) {

      // FORWARD PASS

      // Get batchSize training images and labels
      auto inputData{
          trainingImages->view(batchSequence[batch] * batchSize,
                               (batchSequence[batch] + 1) * batchSize)};
      auto inputCorrect{
          trainingLabels->view(batchSequence[batch] * batchSize,
                               (batchSequence[batch] + 1) * batchSize)};

      dense1->forward(inputData);
      activation1->forward(dense1->output());
      dropout1->forward(activation1->output());
      dense2->forward(dropout1->output());
      activation2->forward(dense2->output());
      dropout2->forward(activation2->output());
      outputLayer->forward(dropout2->output());
      outputSoftmaxLoss->forward(outputLayer->output(), inputCorrect);

      float dataLoss{outputSoftmaxLoss->mean()};
      float regularizationLoss{outputSoftmaxLoss->regularizationLoss(*dense1) +
                               outputSoftmaxLoss->regularizationLoss(*dense2)};
      float loss{dataLoss + regularizationLoss};

      // BACKWARD PASS
      outputSoftmaxLoss->backward();
      outputLayer->backward(outputSoftmaxLoss->dinputs());
      dropout2->backward(outputLayer->dinputs());
      activation2->backward(dropout2->dinputs());
      dense2->backward(activation2->dinputs());
      dropout1->backward(dense2->dinputs());
      activation1->backward(dropout1->dinputs());
      dense1->backward(activation1->dinputs());

      optimizer->preUpdate();
      optimizer->updateParams(*outputLayer);
      optimizer->updateParams(*dense2);
      optimizer->updateParams(*dense1);
      optimizer->postUpdate();

      // Display information every about half second, or at the first/final
      // batch
      if (displayTimer.elapsed() >= 0.5 ||
          batch + 1 == trainingSize / batchSize || batch == 0) {
        std::cout << '\r' << batch + 1 << '/' << trainingSize / batchSize
                  << '\t' << static_cast<size_t>(trainingTimer.elapsed())
                  << "s " << formatTime(trainingTimer.elapsed() / (batch + 1))
                  << "/step \taccuracy: " << outputSoftmaxLoss->accuracy()
                  << " - loss: " << loss << " (data loss: " << dataLoss
                  << ", reg loss: " << regularizationLoss
                  << ") - lr: " << optimizer->learningRate()
                  << "                 " << std::flush;

        displayTimer.reset();
      }
    }

    std::cout << '\n';
  }

  dense1->forward(testingImages);
  activation1->forward(dense1->output());
  dense2->forward(activation1->output());
  activation2->forward(dense2->output());
  outputLayer->forward(activation2->output());
  outputSoftmaxLoss->forward(outputLayer->output(), testingLabels);

  std::cout << "Test loss: " << outputSoftmaxLoss->mean()
            << "\nTest accuracy: " << outputSoftmaxLoss->accuracy() << '\n';
}
