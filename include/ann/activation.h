#pragma once

#include <functional>
#include <vector>

namespace ANN {
struct Activation {
  enum Type {
    Linear,
    Step,
    ReLU,
    Sigmoid,
  } type;

  std::vector<double> args{};

  Activation(Type t, std::vector<double> a = {})
      : type{t}, args{std::move(a)} {}

  Activation(Activation &&other) noexcept;

  std::function<double(double)> getForward();

  // Returns function for backpropagation
  // Returned function input - the output of the layer, the derivative of the
  // specific output
  // Returned function return - the derivative in respect to the activation function
  std::function<double(double, double)> getBackward();
};
} // namespace ANN
