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
    LeakyReLU, // Needs one argument, for multiplicator of input
  } type;

  std::vector<float> args{};

  Activation(Type t, std::vector<float> a = {}) : type{t}, args{std::move(a)} {}

  Activation(Activation &&other) noexcept;

  std::function<float(float)> getForward();

  // Returns function for backpropagation
  // Returned function input - the output of the layer, the derivative of the
  // specific output
  // Returned function return - the derivative in respect to the activation
  // function
  std::function<float(float, float)> getBackward();
};
} // namespace ANN
