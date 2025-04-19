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
};
} // namespace ANN
