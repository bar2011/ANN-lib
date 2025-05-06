#include "math/matrixBase.h"
#include "math/random.h"
#include "mnist/loader.h"

#include "ann/layers/categoricalLossSoftmax.h"
#include "ann/layers/dense.h"
#include "ann/optimizers/sgd.h"

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

int main() {
  try {
    auto loader{std::make_unique<MNist::Loader>(
        "data/train-labels-idx1-ubyte", "data/train-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte", "data/t10k-images-idx3-ubyte")};
    std::array<MNist::Loader::DataPair, 2> data{loader->loadData()};

    // rows and columns in each image
    constexpr int rows{28};
    constexpr int cols{28};
    constexpr int layer1Neurons{16};
    constexpr int layer2Neurons{16};
    constexpr int outputNeurons{10};
    constexpr int batchSize{64};
    constexpr int trainingSize{60'000};
    constexpr int epochs{trainingSize / batchSize};

    // print processed images matrices from data training set
    // printMatrixImage(
    //     std::get<1>(data[0])->view(0, batchSize)->reshape(28 * batchSize,
    //     28));

    auto hiddenLayer1{std::make_unique<Layer::Dense<float>>(
        rows * cols, layer1Neurons, batchSize,
        ANN::Activation{ANN::Activation::LeakyReLU, {1e-2}},
        Layer::WeightInit::He)};

    auto hiddenLayer2{std::make_unique<Layer::Dense<double>>(
        layer1Neurons, layer2Neurons, batchSize,
        ANN::Activation{ANN::Activation::LeakyReLU, {1e-2}},
        Layer::WeightInit::He)};

    auto outputLayer{std::make_unique<Layer::Dense<double>>(
        layer2Neurons, outputNeurons, batchSize,
        ANN::Activation{ANN::Activation::Linear}, Layer::WeightInit::He)};

    auto outputSoftmaxLoss{
        std::make_unique<Layer::CategoricalLossSoftmax<double, unsigned char>>(
            outputNeurons, batchSize)};

    auto optimizer{std::make_unique<Optimizers::SGD>(0.01, 0.001)};

    std::vector<size_t> batchSequence(epochs);
    std::iota(batchSequence.begin(), batchSequence.end(), 0);
    std::shuffle(batchSequence.begin(), batchSequence.end(), Math::Random::mt);

    for (size_t epoch{}; epoch < epochs; ++epoch) {
      // FORWARD PASS

      // Get batchSize training images and labels
      auto inputData{
          std::get<1>(data[0])->view(batchSequence[epoch] * batchSize,
                                     (batchSequence[epoch] + 1) * batchSize)};
      auto inputCorrect{
          std::get<0>(data[0])->view(batchSequence[epoch] * batchSize,
                                     (batchSequence[epoch] + 1) * batchSize)};

      hiddenLayer1->forward(inputData);
      hiddenLayer2->forward(hiddenLayer1->output());
      outputLayer->forward(hiddenLayer2->output());
      outputSoftmaxLoss->forward(outputLayer->output(), inputCorrect);

      if (epoch % 100 == 0)
        std::cout << "epoch: " << epoch
                  << "\tloss: " << outputSoftmaxLoss->mean()
                  << "\tacc: " << outputSoftmaxLoss->accuracy()
                  << "\tlr: " << optimizer->learningRate() << '\n';

      // BACKWARD PASS
      outputSoftmaxLoss->backward();
      outputLayer->backward(outputSoftmaxLoss->dinputs());
      hiddenLayer2->backward(outputLayer->dinputs());
      hiddenLayer1->backward(hiddenLayer2->dinputs());

      optimizer->preUpdate();
      optimizer->updateParams(*outputLayer);
      optimizer->updateParams(*hiddenLayer2);
      optimizer->updateParams(*hiddenLayer1);
      optimizer->postUpdate();
    }

    std::cout << "FINAL loss: " << outputSoftmaxLoss->mean()
              << " acc: " << outputSoftmaxLoss->accuracy() << '\n';

    std::cout << "\nFinished\n";
  } catch (std::runtime_error &e) {
    std::cout << "An error occured: " << e.what() << '\n';
  } catch (...) {
    std::cout << "An unknown error occured\n";
  }

  return 0;
}
