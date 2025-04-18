#include "math/matrixBase.h"
#include "mnist/loader.h"

#include "ann/layers/dense.h"

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
      std::cout << "\033[38;2;" << static_cast<unsigned int>(m[row, col]) << ';'
                << static_cast<unsigned int>(m[row, col]) << ';'
                << static_cast<unsigned int>(m[row, col]) << "m█\033[0m";
    std::cout << '\n';
  }
}

int main() {
  try {
    MNist::Loader *loader{new MNist::Loader(
        "data/train-labels-idx1-ubyte", "data/train-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte", "data/t10k-images-idx3-ubyte")};
    std::array<MNist::Loader::DataPair, 2> data{loader->loadData()};
    delete loader;

    // rows and columns in each image
    constexpr int rows{28};
    constexpr int cols{28};
    constexpr int batchSize{2};

    // print processed images matrices from data training set
    printMatrixImage(std::get<1>(data[0]).view(0, batchSize).reshape(28 * batchSize, 28));

    auto testLayer{new Layer::Dense<unsigned char>(rows * cols, 10, batchSize)};
    // Forward first row of training images
    testLayer->forward(std::get<1>(data[0]).view(0, batchSize));
    auto output{testLayer->output()};

    std::cout << "Rows: " << output.rows() << ", cols: " << output.cols()
              << ", Output: \n";
    for (size_t i{}; i < output.rows(); ++i) {
      for (size_t j{}; j < output.cols(); ++j)
        std::cout << output[i, j] << ' ';
      std::cout << '\n';
    }

    std::cout << "\nFinished\n";
    delete testLayer;
  } catch (std::runtime_error &e) {
    std::cout << "An error occured: " << e.what() << '\n';
  } catch (...) {
    std::cout << "An unknown error occured\n";
  }

  return 0;
}
